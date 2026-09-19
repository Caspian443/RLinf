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

from __future__ import annotations

import pathlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils._pytree import tree_map


def _to_numpy(value: Any) -> Any:
    return np.asarray(value.detach().cpu()) if torch.is_tensor(value) else value


class OpenPITransformRunner:
    """Apply one OpenPI input/output transform pipeline to batched RLinf data."""

    def __init__(
        self,
        input_transforms: Sequence,
        output_transforms: Sequence,
        *,
        config_name: str,
        state_indices: Sequence[int] | None,
        action_chunk: int | None,
        device: torch.device | str,
    ) -> None:
        from openpi.transforms import compose

        self.input_transform_fn = compose(input_transforms)
        self.output_transform_fn = compose(output_transforms)
        self.config_name = config_name
        self.state_indices = list(state_indices) if state_indices else None
        self.action_chunk = action_chunk
        self.device = torch.device(device)

    def select_state(self, states: Any) -> Any:
        """Select the configured proprioceptive dimensions."""
        indices = self.state_indices
        if not indices:
            return states

        state_dim = (
            states.shape[-1]
            if hasattr(states, "shape")
            else np.asarray(states).shape[-1]
        )
        if state_dim == len(indices):
            return states
        if state_dim <= max(indices):
            raise ValueError(
                f"Cannot select state_indices={indices} from state dim {state_dim}."
            )

        if torch.is_tensor(states):
            index_tensor = torch.as_tensor(indices, device=states.device)
            return states.index_select(-1, index_tensor)
        return np.asarray(states)[..., indices]

    def repack_env_obs(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        """Map standardized RLinf observations to OpenPI input keys."""
        env_states = self.select_state(env_obs["states"])
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "calvin" in self.config_name:
            processed_obs["observation/state_ee_pos"] = env_states[:, :3]
            processed_obs["observation/state_ee_rot"] = env_states[:, 3:6]
            processed_obs["observation/state_gripper"] = env_states[:, 6:7]
        else:
            processed_obs["observation/state"] = env_states
        wrist_images = env_obs.get("wrist_images")
        if wrist_images is not None:
            processed_obs["observation/wrist_image"] = wrist_images
        extra_view_images = env_obs.get("extra_view_images")
        if extra_view_images is not None:
            processed_obs["observation/extra_view_image"] = extra_view_images
        return processed_obs

    def input_transform(self, obs: dict[str, Any], transpose: bool = False) -> dict:
        """Apply the configured input transforms independently to each sample."""
        inputs = tree_map(lambda value: value, obs)
        first_process = "prompt" in inputs
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {key: inputs[key] for key in inputs if "/" in key}

        inputs = tree_map(_to_numpy, inputs)
        batch_size = next(
            value.shape[0] for value in inputs.values() if hasattr(value, "shape")
        )
        batch_samples = []
        for index in range(batch_size):
            sample = tree_map(lambda value: value[index], inputs)
            if transpose:
                sample = tree_map(
                    lambda value: (
                        value.transpose(1, 2, 0)
                        if isinstance(value, np.ndarray) and value.ndim == 3
                        else value
                    ),
                    sample,
                )
            if first_process:
                prompts = obs["prompt"]
                if isinstance(prompts, np.ndarray):
                    prompts = prompts.tolist()
                sample["prompt"] = prompts[index]
            else:
                # Tokenize still runs, but the cached tokens below replace its
                # output, so the placeholder text is not observed by the model.
                sample["prompt"] = "xxxx"
            batch_samples.append(sample)

        with ThreadPoolExecutor(max_workers=min(len(batch_samples), 8)) as executor:
            transformed = list(executor.map(self.input_transform_fn, batch_samples))

        recombined = tree_map(
            lambda *values: torch.from_numpy(np.asarray(values).copy()),
            *transformed,
        )
        if not first_process:
            recombined["tokenized_prompt"] = obs["tokenized_prompt"]
            recombined["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return recombined

    def output_transform(self, outputs: dict[str, Any]) -> dict:
        """Apply the configured output transforms independently to each sample."""
        batch_size = outputs["actions"].shape[0]
        transformed = []
        for index in range(batch_size):
            sample = tree_map(
                lambda value: _to_numpy(value[index]),
                outputs,
            )
            transformed.append(self.output_transform_fn(sample))
        recombined = tree_map(
            lambda *values: torch.from_numpy(np.asarray(values).copy()),
            *transformed,
        )
        if self.action_chunk is not None:
            recombined["actions"] = recombined["actions"][:, : self.action_chunk]
        return recombined

    def observation_to_device(
        self,
        processed: dict[str, Any],
        *,
        device: torch.device | str | None = None,
    ) -> Any:
        """Build the canonical OpenPI observation on the selected device."""
        from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation

        observation = Observation.from_dict(processed)
        target_device = self.device if device is None else torch.device(device)

        def _move(value: Any) -> Any:
            return value.to(target_device) if isinstance(value, torch.Tensor) else value

        def _move_state(value: Any) -> Any:
            # The openpi Normalize stage runs in float64 (its norm_stats arrays
            # are float64); cast state back to float32 to match the legacy eval
            # processor (which did a final ``.float()``) and the model's compute
            # dtype. For pi05 the continuous state is unused (only the discrete
            # state tokens in the prompt are), but pi0 feeds it through
            # ``state_proj`` so float32 keeps the linear layer's dtype aligned.
            if isinstance(value, torch.Tensor):
                return value.to(device=target_device, dtype=torch.float32)
            return value

        return Observation(
            images={key: _move(value) for key, value in observation.images.items()},
            image_masks={
                key: _move(value) for key, value in observation.image_masks.items()
            },
            state=_move_state(observation.state),
            tokenized_prompt=_move(observation.tokenized_prompt),
            tokenized_prompt_mask=_move(observation.tokenized_prompt_mask),
            token_ar_mask=_move(observation.token_ar_mask),
            token_loss_mask=_move(observation.token_loss_mask),
            pcd_xyz=_move(observation.pcd_xyz),
        )


def build_openpi_transforms(
    model_path: str,
    config_name: str,
    data_kwargs: dict[str, Any] | None = None,
    *,
    norm_stats_dir: str | None = None,
    norm_stats_asset_id: str | None = None,
) -> tuple[Sequence, Sequence]:
    """Build ``(input_transforms, output_transforms)`` for ``config_name``.

    Returns two lists ready for :func:`openpi.transforms.compose`, matching
    ``rlinf/models/embodiment/openpi/__init__.py`` exactly:

    * input:  ``[InjectDefaultPrompt(None), *data.inputs, Normalize, *model.inputs]``
    * output: ``[*model.outputs, Unnormalize, *data.outputs]``

    Norm stats resolve in this priority order: ``data_config.norm_stats``,
    ``data_kwargs["norm_stats_path"]``, ``norm_stats_dir``, then the downloaded
    checkpoint directory. The BEHAVIOR SFT loader passes the experiment's
    ``assets_dir`` + ``asset_id`` so it reads the exact same ``norm_stats.json``
    the old SFT path did (the SFT *base* checkpoint bundles no stats).
    """
    import openpi.shared.download as download
    import openpi.transforms as transforms
    from openpi.training import checkpoints as _checkpoints

    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    train_config = get_openpi_config(
        config_name, model_path=str(model_path), data_kwargs=data_kwargs
    )
    upstream_model_config = train_config.model

    data_config = train_config.data.create(
        train_config.assets_dirs, upstream_model_config
    )

    explicit_norm_stats_path = (
        data_kwargs.get("norm_stats_path")
        if data_kwargs is not None and data_kwargs.get("norm_stats_path") is not None
        else None
    )
    if explicit_norm_stats_path is not None:
        norm_dir = pathlib.Path(explicit_norm_stats_path).expanduser()
        if norm_dir.is_file():
            norm_dir = norm_dir.parent
        norm_stats_dir = str(norm_dir.parent)
        norm_stats_asset_id = norm_dir.name

    asset_id = norm_stats_asset_id or data_config.asset_id
    if asset_id is None:
        raise ValueError("asset_id is required to load norm_stats.")
    norm_stats = data_config.norm_stats
    if norm_stats is None:
        if norm_stats_dir is not None:
            stats_dir = norm_stats_dir
        elif explicit_norm_stats_path is not None:
            norm_stats_path = pathlib.Path(explicit_norm_stats_path).expanduser()
            norm_stats_dir_path = (
                norm_stats_path.parent if norm_stats_path.is_file() else norm_stats_path
            )
            stats_dir = str(norm_stats_dir_path.parent)
        else:
            stats_dir = download.maybe_download(str(model_path))
        norm_stats = _checkpoints.load_norm_stats(stats_dir, asset_id)
    if norm_stats is None:
        raise FileNotFoundError(
            f"openpi_rlinf: norm_stats not found at {stats_dir}/{asset_id}/"
            "norm_stats.json. For eval/RL the checkpoint dir must bundle them; "
            "for SFT set actor.model.openpi.assets_dir/asset_id to the stats dir."
        )

    input_transforms = [
        transforms.InjectDefaultPrompt(None),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ]
    output_transforms = [
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ]
    return input_transforms, output_transforms


def build_openpi_transforms_from_model_config(
    model_cfg: Any,
) -> tuple[str, Sequence, Sequence]:
    """Resolve the OpenPI transform pipeline from RLinf's model config."""
    from omegaconf import OmegaConf

    openpi_cfg = model_cfg.get("openpi", {})
    config_name = str(openpi_cfg.get("config_name", ""))
    if not config_name:
        raise ValueError(
            "actor.model.openpi.config_name is required for OpenPI preprocessing."
        )

    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    input_transforms, output_transforms = build_openpi_transforms(
        model_cfg.model_path,
        config_name,
        data_kwargs=data_kwargs,
    )
    return config_name, input_transforms, output_transforms
