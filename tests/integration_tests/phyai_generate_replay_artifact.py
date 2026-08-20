"""Generate a real PhyAI PPO trajectory for native Actor replay.

This is an explicit single-GPU smoke helper, not a pytest test.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import torch

from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request, PI05RolloutConfig
from phyai.utils import load_config
from phyai.weights import WeightLoadSession
from phyai_utils_tools.models.pi05 import PI05Processor


def _camera(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"Expected BHWC image batch, got {tuple(image.shape)}.")
    return image.permute(0, 3, 1, 2).float().div(255.0).contiguous()


def _dataset_stats(raw_stats: dict, state_dim: int) -> dict:
    return {
        "observation.state": {
            key: value[:state_dim] for key, value in raw_stats["state"].items()
        },
        "action": raw_stats["actions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--native-preprocess", required=True, type=Path)
    parser.add_argument("--value-head", required=True, type=Path)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--target-keys", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    args = parser.parse_args()

    torch.manual_seed(4321)
    source = torch.load(
        args.native_preprocess, map_location="cpu", weights_only=False
    )
    env = source["env"]
    raw_stats = source["stats"]
    batch_size = int(env["states"].shape[0])
    state_dim = int(env["states"].shape[-1])

    plugin_cfg = replace(
        load_config(args.checkpoint, PI05Config),
        chunk_size=5,
        num_inference_steps=3,
    )
    engine = Engine(
        EngineArgs(
            plugin="pi05",
            plugin_args=PI05Args(
                checkpoint_dir=args.checkpoint,
                config=plugin_cfg,
                max_batch_size=batch_size,
                vision_params_dtype=torch.float32,
                inputs_image_shape=[[224, 224, 3], [224, 224, 3]],
                add_value_head=True,
                require_full_hot_update=False,
            ),
            config=EngineConfig(
                device=DeviceConfig(target="cuda", params_dtype=torch.bfloat16),
                runtime=RuntimeConfig(use_cuda_graph=False),
            ),
        )
    )
    try:
        value_head = torch.load(
            args.value_head, map_location="cpu", weights_only=True
        )
        engine.begin_weight_update()
        engine.update_weights(value_head)
        report = engine.finish_weight_update(version=0)
        if sorted(report.loaded) != sorted(value_head):
            raise RuntimeError(f"Value-head hot update mismatch: {report.summary()}")

        target_session = WeightLoadSession(engine.entry.model)
        target_metadata = {
            name: {"shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, (parameter, _shard, _loader) in target_session.index.items()
        }
        args.target_keys.write_text(
            json.dumps(target_metadata, indent=2, sort_keys=True)
        )

        processor = PI05Processor(
            image_size=plugin_cfg.vision.image_size,
            num_channels=plugin_cfg.vision.num_channels,
            num_images=2,
            tokenizer_max_length=plugin_cfg.tokenizer_max_length,
            action_dim=7,
            tokenizer_name=args.tokenizer,
            dataset_stats=_dataset_stats(raw_stats, state_dim),
            normalize_pixels=True,
            include_state_in_prompt=False,
            device="cuda",
            params_dtype=torch.bfloat16,
        )
        processed = processor.preprocess(
            {
                "images": [_camera(env["main_images"]), _camera(env["wrist_images"])],
                "task": env["task_descriptions"],
                "state": env["states"],
            }
        )
        noise = torch.randn(batch_size, 5, 32, dtype=torch.float32)
        request = PI05Request(
            pixel_values=processed.pixel_values,
            input_ids=processed.input_ids,
            lang_lens=processed.lang_lens,
            noise=noise,
        )
        rollout = engine.rollout_step(
            request,
            rollout_config=PI05RolloutConfig(
                action_chunk=5,
                action_dim=7,
                noise_method="flow_sde",
                noise_level=0.5,
                denoise_index=1,
            ),
        )
        actions = processor.postprocess(rollout.actions).float().contiguous()
        token_mask = (
            torch.arange(processed.input_ids.shape[1], device="cuda")[None, :]
            < processed.lang_lens[:, None]
        )
        forward_inputs = {
            "chains": rollout.chains.detach().cpu(),
            "denoise_inds": rollout.denoise_inds.detach().cpu(),
            "tokenized_prompt": processed.input_ids.detach().cpu(),
            "tokenized_prompt_mask": token_mask.detach().cpu(),
            "action": actions.reshape(batch_size, -1),
            "model_action": rollout.actions.detach().cpu().reshape(batch_size, -1),
            "observation/image": env["main_images"],
            "observation/wrist_image": env["wrist_images"],
            "observation/state": env["states"],
        }
        artifact = {
            "forward_inputs": forward_inputs,
            "prev_logprobs": rollout.prev_logprobs.detach().cpu(),
            "prev_values": rollout.prev_values.detach().cpu(),
            "actions": actions,
            "engine_version": engine.version,
        }
        args.trajectory.parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, args.trajectory)
        evidence = {
            "python": __import__("sys").executable,
            "transformers": __import__("transformers").__version__,
            "engine_version": engine.version,
            "loaded_value_head": len(report.loaded),
            "target_key_count": len(target_metadata),
            "actions_shape": list(actions.shape),
            "chains_shape": list(rollout.chains.shape),
            "denoise_inds": rollout.denoise_inds.tolist(),
            "prev_logprobs_shape": list(rollout.prev_logprobs.shape),
            "prev_logprobs_finite": bool(torch.isfinite(rollout.prev_logprobs).all()),
            "prev_values_shape": list(rollout.prev_values.shape),
            "prev_values_finite": bool(torch.isfinite(rollout.prev_values).all()),
        }
        print(json.dumps(evidence, indent=2, sort_keys=True))
    finally:
        engine.close()


if __name__ == "__main__":
    main()
PATCH
