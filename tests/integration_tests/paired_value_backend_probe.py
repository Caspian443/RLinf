# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compare HF and PhyAI values on one fixed observation sequence.

This is an explicit GPU experiment helper, not a pytest test. Run the ``hf``
command in the OpenPI runtime, ``phyai`` in the PhyAI runtime, then ``compare``
in either runtime. The output artifacts contain tensors only; no model weights
other than the small runtime-initialized value head are duplicated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch


def _model_config(checkpoint: str, norm_stats: str):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "model_type": "openpi_rlinf",
            "model_path": checkpoint,
            "precision": "bf16",
            "pi05": True,
            "num_action_chunks": 5,
            "action_dim": 7,
            "num_steps": 3,
            "add_value_head": True,
            "use_proprio": True,
            "is_lora": False,
            "lora_rank": 32,
            "openpi_data": {"norm_stats_path": norm_stats},
            "openpi": {
                "task": "rl",
                "config_name": "pi05_libero",
                "model_action_horizon": 10,
                "model_action_dim": 32,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "max_token_len": 200,
                "num_images_in_input": 2,
                "action_chunk": 5,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "noise_level": 0.5,
                "train_expert_only": True,
                "value_after_vlm": True,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
                "joint_logprob": False,
                "ignore_last": True,
                "safe_get_logprob": False,
            },
        }
    )


def _base_observations() -> dict[str, Any]:
    """Create two deterministic LIBERO-shaped observations."""
    generator = torch.Generator().manual_seed(20260829)
    return {
        "states": torch.tensor(
            [
                [0.05, -0.10, 0.15, -0.20, 0.25, -0.30, 0.35, -0.40],
                [-0.12, 0.18, -0.24, 0.30, -0.36, 0.42, -0.48, 0.54],
            ],
            dtype=torch.float32,
        ),
        "main_images": torch.randint(
            0, 256, (2, 224, 224, 3), dtype=torch.uint8, generator=generator
        ),
        "wrist_images": torch.randint(
            0, 256, (2, 224, 224, 3), dtype=torch.uint8, generator=generator
        ),
        "task_descriptions": [
            "pick up the red block",
            "put the bowl on the plate",
        ],
    }


def _fixed_sequence(env: dict[str, Any], sequence_steps: int = 3) -> dict[str, Any]:
    """Expand two fixed observations into a time-major sequence."""
    states = []
    main_images = []
    wrist_images = []
    tasks: list[str] = []
    for step in range(sequence_steps):
        states.append(env["states"] + step * 0.01)
        main_images.append(torch.roll(env["main_images"], shifts=step, dims=2))
        wrist_images.append(torch.roll(env["wrist_images"], shifts=-step, dims=1))
        tasks.extend(env["task_descriptions"])
    return {
        "states": torch.cat(states).contiguous(),
        "main_images": torch.cat(main_images).contiguous(),
        "wrist_images": torch.cat(wrist_images).contiguous(),
        "task_descriptions": tasks,
    }


def _value_head_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith("value_head.")
    }


def _tensor_digest(tensor: torch.Tensor) -> str:
    data = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def _canonical_hf_inputs(
    forward_inputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    keys = (
        "obs_state",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "obs_image__base_0_rgb",
        "obs_image__left_wrist_0_rgb",
        "obs_image__right_wrist_0_rgb",
        "obs_image_mask__base_0_rgb",
        "obs_image_mask__left_wrist_0_rgb",
        "obs_image_mask__right_wrist_0_rgb",
    )
    return {key: forward_inputs[key].detach().cpu() for key in keys}


def run_hf(args: argparse.Namespace) -> None:
    from rlinf.models.embodiment.openpi_rlinf import get_model
    from rlinf.utils.utils import seed_everything

    seed_everything(args.seed)
    base_observations = _base_observations()
    env = _fixed_sequence(base_observations)
    args.fixed_observations.parent.mkdir(parents=True, exist_ok=True)
    torch.save(env, args.fixed_observations)

    model = get_model(_model_config(args.actor_checkpoint, args.norm_stats))
    value_head = _value_head_state(model)
    if len(value_head) != 8:
        raise RuntimeError(f"Expected 8 value-head tensors, got {len(value_head)}.")
    torch.save(value_head, args.value_head)

    model = model.to("cuda").eval()
    batch_size = int(env["states"].shape[0])
    generator = torch.Generator(device="cuda").manual_seed(args.noise_seed)
    noise = torch.randn(
        batch_size,
        10,
        32,
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    with torch.no_grad():
        _, rollout = model.predict_action_batch(
            env,
            mode="train",
            compute_values=True,
            noise=noise,
            rng=generator,
        )
        replay = model.default_forward(rollout["forward_inputs"], compute_values=True)

    num_envs = len(base_observations["task_descriptions"])
    sequence_steps = batch_size // num_envs
    result = {
        "backend": "hf",
        "num_envs": num_envs,
        "sequence_steps": sequence_steps,
        "prev_values": rollout["prev_values"]
        .detach()
        .cpu()
        .reshape(sequence_steps, num_envs),
        "actor_values": replay["values"]
        .detach()
        .cpu()
        .reshape(sequence_steps, num_envs),
        "canonical_inputs": _canonical_hf_inputs(rollout["forward_inputs"]),
        "noise": noise.detach().cpu(),
        "value_head_digests": {
            name: _tensor_digest(tensor) for name, tensor in value_head.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "prev_values": result["prev_values"].tolist(),
                "actor_values": result["actor_values"].tolist(),
                "value_head_tensors": len(value_head),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _camera(image: torch.Tensor) -> torch.Tensor:
    return image.permute(0, 3, 1, 2).float().div(255.0).contiguous()


def _load_dataset_stats(path: Path, state_dim: int) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    stats = payload.get("norm_stats", payload)
    return {
        "observation.state": {
            key: values[:state_dim] for key, values in stats["state"].items()
        },
        "action": stats["actions"],
    }


def run_phyai(args: argparse.Namespace) -> None:
    from phyai.engine import Engine, EngineArgs
    from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
    from phyai.models.pi05.configuration_pi05 import PI05Config
    from phyai.models.pi05.main_pi05 import PI05Args
    from phyai.models.pi05.scheduler_ws1_pi05 import PI05RolloutRequest
    from phyai.utils import load_config
    from phyai_utils_tools.models.pi05 import PI05Processor
    from safetensors import safe_open

    env = torch.load(args.fixed_observations, map_location="cpu", weights_only=True)
    hf = torch.load(args.hf_output, map_location="cpu", weights_only=True)
    value_head = torch.load(args.value_head, map_location="cpu", weights_only=True)
    batch_size = int(env["states"].shape[0])
    state_dim = int(env["states"].shape[-1])

    plugin_cfg = replace(
        load_config(args.config_checkpoint, PI05Config),
        chunk_size=10,
        num_inference_steps=3,
        config_name="pi05_libero",
        action_chunk=5,
        add_value_head=True,
        value_after_vlm=True,
        value_vlm_mode="mean_token",
        detach_critic_input=True,
    )
    engine = Engine(
        EngineArgs(
            plugin="pi05",
            plugin_args=PI05Args(
                checkpoint_dir=None,
                config=plugin_cfg,
                max_batch_size=batch_size,
                inputs_image_shape=[[224, 224, 3], [224, 224, 3]],
                capture_rollout=True,
                require_full_hot_update=True,
                defer_scheduler_setup=True,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(
                    use_cuda_graph=True,
                    flashinfer_workspace_bytes=4 * 1024**3,
                ),
            ),
        )
    )
    try:
        engine.begin_weight_update()
        with safe_open(
            str(args.actor_checkpoint), framework="pt", device="cpu"
        ) as checkpoint:
            engine.update_weights(
                (name, checkpoint.get_tensor(name)) for name in checkpoint.keys()
            )
        engine.update_weights(value_head)
        report = engine.finish_weight_update(version=0)
        if report.missing or report.unexpected:
            raise RuntimeError(f"Strict Actor update mismatch: {report.summary()}")

        processor = PI05Processor(
            image_size=plugin_cfg.vision.image_size,
            num_channels=plugin_cfg.vision.num_channels,
            num_images=2,
            tokenizer_max_length=plugin_cfg.tokenizer_max_length,
            action_dim=7,
            tokenizer_name=args.tokenizer,
            dataset_stats=_load_dataset_stats(args.norm_stats, state_dim),
            normalize_pixels=True,
            include_state_in_prompt=False,
            device="cuda",
            params_dtype=torch.bfloat16,
        )
        processed = processor.preprocess(
            {
                "images": [_camera(env["main_images"]), _camera(env["wrist_images"])],
                "task": env["task_descriptions"],
                "state": env["states"].to("cuda"),
            }
        )
        token_ids = processed.input_ids.to("cuda")
        token_mask = (
            torch.arange(token_ids.shape[1], device="cuda")[None, :]
            < processed.lang_lens.to("cuda")[:, None]
        )
        rollout = engine.rollout_step(
            PI05RolloutRequest(
                pixel_values=processed.pixel_values.to("cuda", dtype=torch.bfloat16),
                input_ids=token_ids,
                lang_lens=processed.lang_lens.to("cuda"),
                noise=hf["noise"].to("cuda"),
                noise_method="flow_sde",
                noise_level=0.5,
                action_chunk=5,
                action_dim=7,
                joint_logprob=False,
                ignore_last=True,
                safe_get_logprob=False,
                compute_values=True,
            )
        )
        state = processed.state.to("cuda", dtype=torch.float32)
        state = torch.nn.functional.pad(state, (0, 32 - state.shape[-1]))
        image_mask = torch.ones(batch_size, dtype=torch.bool, device="cuda")
        images = processed.pixel_values.to("cuda")
        canonical_inputs = {
            "obs_state": state,
            "tokenized_prompt": token_ids,
            "tokenized_prompt_mask": token_mask,
            "obs_image__base_0_rgb": images[:, 0].permute(0, 2, 3, 1),
            "obs_image__left_wrist_0_rgb": images[:, 1].permute(0, 2, 3, 1),
            "obs_image__right_wrist_0_rgb": torch.zeros_like(images[:, 0]).permute(
                0, 2, 3, 1
            ),
            "obs_image_mask__base_0_rgb": image_mask,
            "obs_image_mask__left_wrist_0_rgb": image_mask,
            "obs_image_mask__right_wrist_0_rgb": torch.zeros_like(image_mask),
        }
        result = {
            "backend": "phyai",
            "num_envs": hf["num_envs"],
            "sequence_steps": hf["sequence_steps"],
            "prev_values": rollout.prev_values.detach()
            .cpu()
            .reshape(hf["sequence_steps"], hf["num_envs"]),
            "canonical_inputs": {
                key: value.detach().cpu() for key, value in canonical_inputs.items()
            },
            "loaded_weights": len(report.loaded),
            "missing_weights": len(report.missing),
            "unexpected_weights": len(report.unexpected),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(result, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "prev_values": result["prev_values"].tolist(),
                    "loaded_weights": result["loaded_weights"],
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        engine.close()


def _delta_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    delta = left.float() - right.float()
    return {
        "left": left.float().tolist(),
        "right": right.float().tolist(),
        "mean_signed_delta": float(delta.mean()),
        "mean_abs_delta": float(delta.abs().mean()),
        "max_abs_delta": float(delta.abs().max()),
    }


def compare(args: argparse.Namespace) -> None:
    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
    from rlinf.algorithms.losses import compute_ppo_critic_loss
    from rlinf.utils.metric_utils import (
        compute_critic_explained_variance_from_stats,
    )

    hf = torch.load(args.hf_output, map_location="cpu", weights_only=True)
    phyai = torch.load(args.phyai_output, map_location="cpu", weights_only=True)
    if (hf["num_envs"], hf["sequence_steps"]) != (
        phyai["num_envs"],
        phyai["sequence_steps"],
    ):
        raise RuntimeError("HF and PhyAI artifacts have different sequence geometry.")

    num_envs = int(hf["num_envs"])
    rewards = torch.zeros(hf["sequence_steps"] - 1, num_envs)
    rewards[-1, 0] = 1.0
    dones = torch.zeros(hf["sequence_steps"], num_envs, dtype=torch.bool)
    dones[-1] = True
    loss_mask = torch.ones_like(rewards, dtype=torch.bool)
    loss_mask_sum = torch.full_like(rewards, float(rewards.shape[0]))

    backend_results: dict[str, Any] = {}
    returns_by_backend: dict[str, torch.Tensor] = {}
    actor_values = hf["actor_values"][:-1].float()
    for name, artifact in (("hf", hf), ("phyai", phyai)):
        prev_values = artifact["prev_values"].float()
        _, returns = compute_gae_advantages_and_returns(
            rewards=rewards,
            values=prev_values,
            dones=dones,
            gamma=0.99,
            gae_lambda=0.95,
            normalize_advantages=False,
            loss_mask=loss_mask,
        )
        value_loss, metrics = compute_ppo_critic_loss(
            values=actor_values,
            returns=returns,
            prev_values=prev_values[:-1],
            value_clip=0.2,
            huber_delta=10.0,
            loss_mask=loss_mask,
            loss_mask_sum=loss_mask_sum,
            max_episode_steps=10,
        )
        returns_by_backend[name] = returns
        backend_results[name] = {
            "returns": returns.tolist(),
            "returns_mean": float(returns.mean()),
            "value_loss": float(value_loss),
            "explained_variance": float(
                compute_critic_explained_variance_from_stats(metrics)
            ),
        }

    input_stats = {}
    for key, hf_tensor in hf["canonical_inputs"].items():
        phyai_tensor = phyai["canonical_inputs"][key]
        if hf_tensor.dtype == torch.bool or not hf_tensor.is_floating_point():
            input_stats[key] = {
                "equal": bool(torch.equal(hf_tensor, phyai_tensor)),
                "different_elements": int((hf_tensor != phyai_tensor).sum()),
            }
        else:
            input_stats[key] = _delta_stats(hf_tensor, phyai_tensor)
            input_stats[key].pop("left")
            input_stats[key].pop("right")

    value_loss_abs_delta = abs(
        backend_results["hf"]["value_loss"] - backend_results["phyai"]["value_loss"]
    )
    value_loss_relative_delta = value_loss_abs_delta / abs(
        backend_results["hf"]["value_loss"]
    )

    evidence = {
        "geometry": {
            "num_envs": num_envs,
            "sequence_steps": int(hf["sequence_steps"]),
            "paired_value_samples": int(hf["prev_values"].numel()),
            "loss_samples": int(rewards.numel()),
        },
        "rewards": rewards.tolist(),
        "dones": dones.tolist(),
        "prev_values": _delta_stats(hf["prev_values"], phyai["prev_values"]),
        "returns": _delta_stats(returns_by_backend["hf"], returns_by_backend["phyai"]),
        "actor_values": actor_values.tolist(),
        "backend_results": backend_results,
        "value_loss_abs_delta": value_loss_abs_delta,
        "value_loss_relative_delta": value_loss_relative_delta,
        "canonical_input_stats": input_stats,
        "phyai_weight_update": {
            "loaded": phyai["loaded_weights"],
            "missing": phyai["missing_weights"],
            "unexpected": phyai["unexpected_weights"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    hf_parser = subparsers.add_parser("hf")
    hf_parser.add_argument("--actor-checkpoint", required=True)
    hf_parser.add_argument("--norm-stats", required=True)
    hf_parser.add_argument("--fixed-observations", required=True, type=Path)
    hf_parser.add_argument("--value-head", required=True, type=Path)
    hf_parser.add_argument("--output", required=True, type=Path)
    hf_parser.add_argument("--seed", type=int, default=42)
    hf_parser.add_argument("--noise-seed", type=int, default=20260829)
    hf_parser.set_defaults(func=run_hf)

    phyai_parser = subparsers.add_parser("phyai")
    phyai_parser.add_argument("--actor-checkpoint", required=True, type=Path)
    phyai_parser.add_argument("--config-checkpoint", required=True)
    phyai_parser.add_argument("--norm-stats", required=True, type=Path)
    phyai_parser.add_argument("--tokenizer", required=True)
    phyai_parser.add_argument("--fixed-observations", required=True, type=Path)
    phyai_parser.add_argument("--value-head", required=True, type=Path)
    phyai_parser.add_argument("--hf-output", required=True, type=Path)
    phyai_parser.add_argument("--output", required=True, type=Path)
    phyai_parser.set_defaults(func=run_phyai)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--hf-output", required=True, type=Path)
    compare_parser.add_argument("--phyai-output", required=True, type=Path)
    compare_parser.add_argument("--output", required=True, type=Path)
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
