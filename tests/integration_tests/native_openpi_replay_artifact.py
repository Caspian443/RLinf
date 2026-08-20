"""Prepare and replay a PhyAI trajectory with the native OpenPI Actor.

This is an explicit GPU smoke helper, not a pytest test. Run prepare before
the PhyAI artifact generator, then replay after it.
"""

from __future__ import annotations

import argparse
import types
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi import get_model


def _model_config(checkpoint: str):
    return OmegaConf.create(
        {
            "model_type": "openpi",
            "model_path": checkpoint,
            "precision": None,
            "num_action_chunks": 5,
            "action_dim": 7,
            "num_steps": 3,
            "add_value_head": True,
            "use_proprio": True,
            "is_lora": False,
            "lora_rank": 32,
            "openpi": {
                "config_name": "pi05_libero",
                "num_images_in_input": 2,
                "noise_level": 0.5,
                "action_chunk": 5,
                "num_steps": 3,
                "train_expert_only": True,
                "action_env_dim": 7,
                "noise_method": "flow_sde",
                "add_value_head": True,
                "value_after_vlm": True,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
                "joint_logprob": False,
                "ignore_last": False,
                "safe_get_logprob": False,
            },
        }
    )


def _build_model(checkpoint: str):
    torch.manual_seed(1234)
    model = get_model(_model_config(checkpoint))
    return model


def _value_head_state(model) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith("value_head.")
    }


def prepare(args: argparse.Namespace) -> None:
    model = _build_model(args.checkpoint)
    state = model.state_dict()
    value_head = _value_head_state(model)
    if len(value_head) != 8:
        raise RuntimeError(f"Expected 8 value-head tensors, got {len(value_head)}.")

    args.value_head.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value_head, args.value_head)
    metadata = {
        name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)}
        for name, tensor in state.items()
    }
    args.actor_keys.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print("actor_parameter_count", len(state))
    print("actor_value_head_count", len(value_head))
    print("actor_python", __import__("sys").executable)
    print("actor_transformers", __import__("transformers").__version__)


def replay(args: argparse.Namespace) -> None:
    artifact = torch.load(args.trajectory, map_location="cpu", weights_only=False)
    model = _build_model(args.checkpoint)
    value_head = torch.load(args.value_head, map_location="cpu", weights_only=True)
    incompatible = model.load_state_dict(value_head, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected value-head keys: {incompatible.unexpected_keys}"
        )
    model = model.to("cuda").eval()

    forward_inputs = artifact["forward_inputs"]
    device_keys = {
        "chains",
        "denoise_inds",
        "tokenized_prompt",
        "tokenized_prompt_mask",
        "action",
        "model_action",
    }
    forward_inputs = {
        key: value.to("cuda") if key in device_keys else value
        for key, value in forward_inputs.items()
    }
    native_velocities = []
    original_sample_mean_var_val = model.sample_mean_var_val

    def capture_velocity(_self, *call_args, **call_kwargs):
        result = original_sample_mean_var_val(*call_args, **call_kwargs)
        native_velocities.append(result[3].detach().float())
        return result

    model.sample_mean_var_val = types.MethodType(capture_velocity, model)

    with torch.no_grad():
        output = model.default_forward(forward_inputs, compute_values=True)
    native_velocity = native_velocities[0]
    phyai_velocity = artifact["phyai_velocity"].to("cuda").float()
    velocity_delta = native_velocity - phyai_velocity

    old_logprob = artifact["prev_logprobs"].to("cuda").float()
    new_logprob = output["logprobs"].float()
    delta = new_logprob - old_logprob
    ratio = torch.exp(delta)
    values = output["values"].float()
    chunk_delta = delta.flatten(1).sum(dim=1)
    chunk_ratio = torch.exp(chunk_delta)
    old_values = artifact["prev_values"].to("cuda").flatten().float()
    evidence = {
        "old_shape": list(old_logprob.shape),
        "new_shape": list(new_logprob.shape),
        "old_finite": bool(torch.isfinite(old_logprob).all()),
        "new_finite": bool(torch.isfinite(new_logprob).all()),
        "values_shape": list(values.shape),
        "values_finite": bool(torch.isfinite(values).all()),
        "mean_logprob_delta": float(delta.mean()),
        "max_abs_logprob_delta": float(delta.abs().max()),
        "ratio_mean": float(ratio.mean()),
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "chunk_logprob_delta": [float(value) for value in chunk_delta],
        "chunk_ratio": [float(value) for value in chunk_ratio],
        "chunk_ratio_mean": float(chunk_ratio.mean()),
        "old_values_finite": bool(torch.isfinite(old_values).all()),
        "value_mean_delta": float((values - old_values).mean()),
        "value_max_abs_delta": float((values - old_values).abs().max()),
        "velocity_shape": list(native_velocity.shape),
        "velocity_native_finite": bool(torch.isfinite(native_velocity).all()),
        "velocity_phyai_finite": bool(torch.isfinite(phyai_velocity).all()),
        "velocity_mean_abs_delta": float(velocity_delta.abs().mean()),
        "velocity_max_abs_delta": float(velocity_delta.abs().max()),
        "velocity_native_abs_mean": float(native_velocity.abs().mean()),
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if not all(
        (
            evidence["old_finite"],
            evidence["new_finite"],
            evidence["values_finite"],
        )
    ):
        raise RuntimeError("Actor replay produced non-finite PPO tensors.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "replay"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--value-head", required=True, type=Path)
    parser.add_argument("--actor-keys", type=Path)
    parser.add_argument("--trajectory", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        if args.actor_keys is None:
            parser.error("prepare requires --actor-keys")
        prepare(args)
    else:
        if args.trajectory is None:
            parser.error("replay requires --trajectory")
        replay(args)


if __name__ == "__main__":
    main()
