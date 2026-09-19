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

from typing import Any, Literal, Sequence

import numpy as np
import torch

from rlinf.models.embodiment.openpi_rlinf.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf.transforms_pipeline import (
    OpenPITransformRunner,
)
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    OpenPiPytorchRLTConfig,
)


class OpenPiPytorchEvalActionModel(OpenPiPytorchActionModel):
    """Eval-capable wrapper around the vendored ``Pi0`` model.

    Drives observation construction through the upstream ``openpi.transforms``
    pipeline: the factory calls :meth:`setup_wrappers` with the composed
    input/output transform lists from :func:`get_openpi_config`, then
    :meth:`predict_action_batch` routes ``env_obs`` through
    :meth:`_repack_env_obs` → :meth:`input_transform` →
    :meth:`_observation_dict_to_device` → :meth:`Pi0.sample_actions` →
    :meth:`output_transform`.

    The RL subclass inherits this eval path unchanged and adds the PPO
    chain-collecting SDE sampler on top.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
        rlt_cfg: OpenPiPytorchRLTConfig | None = None,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            rlt_cfg=rlt_cfg,
        )
        # ``action_chunk`` slices the env-action subspace from the model output
        # in :meth:`output_transform`.
        self.action_chunk = action_chunk
        # ``config_name`` is the openpi TrainConfig key (e.g. ``pi05_behavior``)
        # used by :meth:`_repack_env_obs` to switch the env→``observation/*``
        # repack rules (currently only the ``calvin`` split-state layout).
        self.config_name = config_name
        # Optional subset of the raw env state dim (openpi ``state_indices``).
        # ``None`` (the BEHAVIOR default) is an identity passthrough.
        self.state_indices = list(state_indices) if state_indices else None

        # openpi.transforms pipeline state (installed by :meth:`setup_wrappers`).
        self._transform_runner: OpenPITransformRunner | None = None

    # -------------------------------------------------------- transforms glue

    def setup_wrappers(
        self,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the openpi.transforms input/output pipelines.

        ``transforms`` is the list passed to the openpi-side ``compose`` —
        typically ``BehaviorInputs → Normalize(norm_stats) → ModelTransformFactory(model_config)``
        (the last stage carries the auto-downloading PaliGemma tokenizer).
        ``output_transforms`` is the matching reverse pipeline used to turn
        sampled model actions back into env-frame actions.
        """
        self._transform_runner = OpenPITransformRunner(
            transforms,
            output_transforms,
            config_name=self.config_name,
            state_indices=self.state_indices,
            action_chunk=self.action_chunk,
            device=self.device,
        )

    def _ensure_wrappers(self) -> OpenPITransformRunner:
        if self._transform_runner is None:
            raise RuntimeError(
                f"{type(self).__name__}.setup_wrappers(...) must be called "
                "after construction (the factory does this); the openpi "
                "transforms pipeline is not yet installed."
            )
        return self._transform_runner

    def _select_configured_state(self, states):
        """Select a configured subset of the raw env state (openpi parity).

        Mirrors ``openpi.openpi_action_model.OpenPi0ForRLActionPrediction.
        _select_configured_state``: ``state_indices=None`` (BEHAVIOR) is an
        identity passthrough; a configured index list fancy-indexes the last
        dim so future non-BEHAVIOR envs can reuse the generic repack.
        """
        return self._ensure_wrappers().select_state(states)

    def _repack_env_obs(self, env_obs: dict) -> dict:
        """Map the env's observation dict to the ``observation/*`` keys the
        openpi pipeline expects.

        Generic adapter mirroring
        ``openpi.openpi_action_model.OpenPi0ForRLActionPrediction.obs_processor``:
        it reads the standardized RLinf env keys (``states`` / ``main_images`` /
        ``task_descriptions`` and the optional ``wrist_images`` /
        ``extra_view_images``) and repacks them into openpi's ``observation/*``
        convention. All real per-env logic lives in the config-driven
        ``data_transforms.inputs`` (selected by ``config_name``), not here; the
        only per-env branch is the ``calvin`` split-state layout. Optional
        camera views are added only when the env actually provides them
        (BEHAVIOR, for instance, emits no ``extra_view_images`` key).
        """
        return self._ensure_wrappers().repack_env_obs(env_obs)

    def input_transform(self, obs: dict, transpose: bool = False) -> dict:
        """Apply the openpi input pipeline per-sample then recombine into a batched dict.

        Two modes:

        * Rollout (``"prompt"`` key present) — runs the full pipeline including
          prompt tokenization; result has ``image``/``image_mask``/``state``/
          ``tokenized_prompt``/``tokenized_prompt_mask`` keys.
        * Train recompute (no ``"prompt"``; only ``observation/*`` and the cached
          ``tokenized_prompt`` keys present) — re-runs the pipeline using the
          cached tokens rather than re-tokenising every micro-batch.
        """
        return self._ensure_wrappers().input_transform(obs, transpose=transpose)

    def output_transform(self, outputs: dict) -> dict:
        """Apply the openpi output pipeline per-sample then recombine."""
        return self._ensure_wrappers().output_transform(outputs)

    def _observation_dict_to_device(self, processed: dict) -> Observation:
        """Convert a per-key dict (from :meth:`input_transform`) into a device-resident :class:`Observation`."""
        return self._ensure_wrappers().observation_to_device(
            processed, device=self.device
        )

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample env actions via the deterministic Euler ODE sampler.

        Only ``mode="eval"`` is supported at the eval level — that path is
        shared by every transforms-pipeline variant (eval task + RL eval
        fallback). ``mode="train"`` is implemented in the RL subclass which
        overrides this method to add the chain-collecting SDE sampler.
        Calling with ``mode="train"`` on the eval class raises
        :class:`NotImplementedError` so an eval-only model loudly refuses to
        be used for on-policy rollouts.
        """
        del compute_values, kwargs  # accepted for call-site parity; eval ignores them
        if mode != "eval":
            raise NotImplementedError(
                f"{type(self).__name__} only supports predict_action_batch(mode='eval'); "
                "use the RL subclass (actor.model.openpi.task='rl') for train rollouts."
            )
        # openpi.transforms pipeline (eval / RL).
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        observation = self._observation_dict_to_device(processed)
        return self._predict_eval(observation, noise=noise, rng=rng)

    def _predict_eval(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Deterministic Euler ODE sampler shared by eval and the RL eval path.

        Returns the env-frame ``actions`` plus the rollout-side return dict
        with ``prev_logprobs=None`` / ``prev_values=None`` and a minimal
        ``forward_inputs`` (action + raw model_action) — exactly what
        :class:`huggingface_worker.HuggingFaceWorker.predict` expects from an
        eval call.
        """
        model_actions = self.model.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        env_outputs = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )
        # openpi Unnormalize runs in float64; cast env actions back to float32 to
        # match the legacy eval processor's ``.astype(np.float32)`` contract (and
        # the action dtype the env/rollout worker expects).
        actions = env_outputs["actions"].to(device=self.device, dtype=torch.float32)
        B = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
        }
        return actions, result

    @torch.no_grad()
    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract the frozen Stage1 features consumed by the Stage2 RLT head."""
        self._require_rlt()
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        observation = self._observation_dict_to_device(processed)

        prepared_observation = pi0_model_module.preprocess_observation(
            observation, train=False
        )
        prefix_output, prefix_mask, kv_cache = self.model.build_prefix_cache(
            prepared_observation
        )
        rlt_prefix_output, rlt_prefix_mask = self._select_rlt_prefix_embeddings(
            prefix_output, prefix_mask, prepared_observation.tokenized_prompt
        )
        z_rl = self._encode_rlt_flat(rlt_prefix_output, rlt_prefix_mask).to(
            dtype=torch.float32
        )

        model_actions = self._sample_actions_from_prefix_cache(
            prepared_observation,
            prefix_mask,
            kv_cache,
        )
        ref_chunk = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )["actions"]

        raw_proprio = self._select_configured_state(env_obs["states"])
        if "maniskill" in self.config_name.lower():
            state_dim = (
                raw_proprio.shape[-1]
                if hasattr(raw_proprio, "shape")
                else np.asarray(raw_proprio).shape[-1]
            )
            proprio = observation.state[..., :state_dim]
        else:
            proprio = raw_proprio
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)

        return {
            "z_rl": z_rl,
            "proprio": proprio.to(device=z_rl.device, dtype=torch.float32),
            "ref_chunk": ref_chunk.to(device=z_rl.device, dtype=torch.float32),
        }

    def _sample_actions_from_prefix_cache(
        self,
        observation: Observation,
        prefix_mask: torch.Tensor,
        kv_cache: tuple,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Run Pi0's eval Euler sampler using an already-built prefix cache."""
        batch_size = observation.state.shape[0]
        device = observation.state.device
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.model.action_horizon,
                self.model.action_dim,
                device=device,
                generator=rng,
            )

        x_t = noise
        dt = -1.0 / self.num_steps
        t = 1.0
        while t >= -dt / 2:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            suffix_out = self.model.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.model.velocity_from_suffix(suffix_out)
            x_t = x_t + dt * v_t
            t += dt
        return x_t
