# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PhyAI-backed embodied rollout worker."""

import gc
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import torch
from omegaconf import DictConfig, OmegaConf
from phyai.engine import Engine, EngineArgs
from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
from phyai.models.pi05.configuration_pi05 import PI05Config
from phyai.models.pi05.main_pi05 import PI05Args
from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request, PI05RolloutConfig
from phyai.utils import load_config
from phyai_utils_tools.models.pi05 import PI05Processor

from rlinf.config import torch_dtype_from_precision
from rlinf.hybrid_engines.weight_syncer.bucket_syncer import BucketWeightSyncer
from rlinf.scheduler import Worker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class _PhyAIWeightTarget(torch.nn.Module):
    """Adapt bucket syncer's ``load_state_dict`` calls to a PhyAI Engine."""

    def __init__(self, engine: Engine) -> None:
        super().__init__()
        self.engine = engine

    def load_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        assign: bool = False,
    ) -> Any:
        del strict, assign
        self.engine.update_weights(state_dict)
        return None


class PhyAIWorker(MultiStepRolloutWorker):
    """Run the reusable multi-step rollout loop with a PhyAI engine.

    The RLinf worker remains the Ray actor and owns one in-process PhyAI
    ``Engine``. Channel communication, batch routing, and the evaluation loop
    are inherited from :class:`MultiStepRolloutWorker`; model construction and
    action inference are replaced with the PhyAI engine path.

    Evaluation keeps the original action-only ``Engine.step`` path. Training
    opts into ``Engine.rollout_step`` and returns the native OpenPI replay
    contract, including denoise states, behavior logprobs, and values.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self._phyai_cfg = cfg.rollout.get("phyai", {})
        self._phyai_plugin = str(self._phyai_cfg.get("plugin", "pi05")).lower()
        self._engine = None
        self._processor = None
        self._weight_target = None
        self._engine_device: torch.device | None = None
        self._engine_dtype: torch.dtype | None = None
        self._normalize_pixels = False

        if self._phyai_plugin != "pi05":
            raise NotImplementedError(
                "PhyAIWorker currently supports only the 'pi05' engine plugin; "
                f"got {self._phyai_plugin!r}."
            )
        if self.enable_offload:
            raise NotImplementedError(
                "PhyAIWorker does not support rollout.enable_offload yet."
            )
        if len(self.global_accelerator_ids) != 1:
            raise ValueError(
                "The initial PhyAI integration requires exactly one GPU per "
                "rollout Ray actor (PhyAI world_size=1); got "
                f"{self.global_accelerator_ids}."
            )

    def init_worker(self) -> None:
        """Construct one PhyAI engine and its cached pi0.5 processor."""
        if self._engine is not None:
            raise RuntimeError("PhyAI engine is already initialized.")

        checkpoint_dir = str(
            self._phyai_cfg.get("checkpoint_dir", None) or self.model_cfg.model_path
        )
        plugin_cfg = load_config(checkpoint_dir, PI05Config)
        if not self.only_eval:
            model_action_horizon = self._phyai_cfg.get("model_action_horizon")
            if model_action_horizon is None:
                raise ValueError(
                    "PhyAI training requires rollout.phyai.model_action_horizon; "
                    "it is the native OpenPI model horizon, not num_action_chunks."
                )
            model_action_horizon = int(model_action_horizon)
            if model_action_horizon < int(self.model_cfg.num_action_chunks):
                raise ValueError(
                    f"PhyAI model_action_horizon={model_action_horizon} is smaller "
                    f"than num_action_chunks={self.model_cfg.num_action_chunks}."
                )
            plugin_cfg = replace(
                plugin_cfg,
                chunk_size=model_action_horizon,
                num_inference_steps=int(self.model_cfg.num_steps),
            )
        engine_precision = self._phyai_cfg.get("params_dtype", self.model_cfg.precision)
        if engine_precision is None:
            engine_precision = "bf16"
        dtype = torch_dtype_from_precision(engine_precision)
        if dtype is None:
            raise ValueError(
                f"Unsupported PhyAI model precision: {engine_precision!r}."
            )
        self._engine_dtype = dtype
        self._engine_device = torch.device(self.torch_device_type)

        runtime_cfg = self._phyai_cfg.get("runtime", {})
        runtime_values = (
            OmegaConf.to_container(runtime_cfg, resolve=True)
            if OmegaConf.is_config(runtime_cfg)
            else runtime_cfg
        )
        runtime_values = dict(runtime_values or {})
        requested_cuda_graph = bool(
            self._phyai_cfg.get("use_cuda_graph", self.only_eval)
        )
        if not self.only_eval and requested_cuda_graph:
            raise ValueError(
                "PhyAI PPO rollout requires rollout.phyai.use_cuda_graph=false."
            )
        runtime_values.setdefault("use_cuda_graph", requested_cuda_graph)

        configured_max_batch_size = self._phyai_cfg.get("max_batch_size", None)
        max_batch_size = int(
            configured_max_batch_size
            if configured_max_batch_size is not None
            else max(self.per_node_train_batch_size, self.per_node_eval_batch_size, 1)
        )
        num_images = int(
            self._phyai_cfg.get(
                "num_images",
                self.model_cfg.get("openpi", {}).get("num_images_in_input", 3),
            )
        )
        if not self.only_eval and num_images != 2:
            raise ValueError(
                "Native OpenPI pi0.5 PPO parity currently requires exactly two "
                f"real cameras; got num_images={num_images}."
            )
        vision_dtype = self._optional_dtype(
            self._phyai_cfg.get("vision_params_dtype", None)
        )
        inputs_image_shape = [
            [
                plugin_cfg.vision.image_size,
                plugin_cfg.vision.image_size,
                plugin_cfg.vision.num_channels,
            ]
            for _ in range(num_images)
        ]

        self.log_info(
            "Launching PhyAI engine: "
            f"plugin={self._phyai_plugin}, checkpoint={checkpoint_dir}, "
            f"max_batch_size={max_batch_size}, num_images={num_images}, "
            f"device={self._engine_device}, dtype={dtype}."
        )
        engine = Engine(
            EngineArgs(
                plugin=self._phyai_plugin,
                plugin_args=PI05Args(
                    checkpoint_dir=checkpoint_dir,
                    max_batch_size=max_batch_size,
                    config=plugin_cfg,
                    weight_remap=self._actor_weight_name,
                    vision_params_dtype=vision_dtype,
                    inputs_image_shape=inputs_image_shape,
                    add_value_head=not self.only_eval,
                    require_full_hot_update=not self.only_eval,
                ),
                config=EngineConfig(
                    device=DeviceConfig(
                        target=str(self._engine_device), params_dtype=dtype
                    ),
                    runtime=RuntimeConfig(**runtime_values),
                ),
            )
        )

        self._normalize_pixels = bool(
            self._phyai_cfg.get("normalize_pixels", not self.only_eval)
        )
        processor_kwargs = {
            "image_size": plugin_cfg.vision.image_size,
            "num_channels": plugin_cfg.vision.num_channels,
            "num_images": num_images,
            "action_dim": int(self.model_cfg.action_dim),
            "normalize_pixels": self._normalize_pixels,
            "image_pad_value": float(self._phyai_cfg.get("image_pad_value", 0.0)),
            "device": self._engine_device,
            "params_dtype": dtype,
        }
        try:
            if self.only_eval and self._phyai_cfg.get(
                "processor_from_pretrained", True
            ):
                processor = PI05Processor.from_pretrained(
                    checkpoint_dir, **processor_kwargs
                )
            else:
                processor_options: dict[str, Any] = {}
                if not self.only_eval:
                    state_dim = self._phyai_cfg.get("state_dim")
                    norm_stats_path = self._phyai_cfg.get("norm_stats_path")
                    tokenizer_name = self._phyai_cfg.get("tokenizer_name")
                    if state_dim is None or norm_stats_path is None:
                        raise ValueError(
                            "PhyAI training requires rollout.phyai.state_dim and "
                            "rollout.phyai.norm_stats_path."
                        )
                    if tokenizer_name is None:
                        raise ValueError(
                            "PhyAI training requires rollout.phyai.tokenizer_name."
                        )
                    state_dim = int(state_dim)
                    processor_options.update(
                        dataset_stats=self._load_dataset_stats(
                            norm_stats_path, state_dim
                        ),
                        tokenizer_name=str(tokenizer_name),
                        include_state_in_prompt=False,
                    )
                processor = PI05Processor(
                    tokenizer_max_length=plugin_cfg.tokenizer_max_length,
                    **processor_options,
                    **processor_kwargs,
                )
        except Exception:
            engine.close()
            raise

        self._engine = engine
        self._weight_target = _PhyAIWeightTarget(engine)
        self._processor = processor

    @staticmethod
    def _optional_dtype(value: Any) -> torch.dtype | None:
        if value is None:
            return None
        if isinstance(value, torch.dtype):
            return value
        precision = {
            "bfloat16": "bf16",
            "float16": "fp16",
            "float32": "fp32",
        }.get(str(value).lower(), str(value))
        dtype = torch_dtype_from_precision(precision)
        if dtype is None:
            raise ValueError(f"Unsupported PhyAI vision_params_dtype: {value!r}.")
        return dtype

    @staticmethod
    def _actor_weight_name(name: str) -> str:
        """Map the native wrapper prefix to PhyAI HF parameter names."""
        return name[len("model.") :] if name.startswith("model.") else name

    @staticmethod
    def _load_dataset_stats(
        path: str | Path, state_dim: int
    ) -> dict[str, dict[str, Any]]:
        """Load OpenPI stats and trim state stats before native-style padding."""
        with Path(path).open(encoding="utf-8") as stream:
            payload = json.load(stream)
        stats = payload.get("norm_stats", payload)
        if "state" not in stats or "actions" not in stats:
            raise ValueError(
                f"OpenPI norm stats at {path} must contain state and actions."
            )

        state_stats: dict[str, Any] = {}
        for key, values in stats["state"].items():
            if len(values) < state_dim:
                raise ValueError(
                    f"state stat {key!r} has {len(values)} values, "
                    f"fewer than state_dim={state_dim}."
                )
            state_stats[key] = values[:state_dim]
        return {
            "observation.state": state_stats,
            "action": stats["actions"],
        }

    @staticmethod
    def _camera_batches(value: Any, name: str) -> list[torch.Tensor]:
        """Normalize one RLinf camera field to a list of BCHW tensors."""
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            cameras = list(value)
        elif isinstance(value, torch.Tensor) and value.ndim == 5:
            cameras = [value[:, index] for index in range(value.shape[1])]
        else:
            cameras = [value]

        normalized = []
        for index, camera in enumerate(cameras):
            if not isinstance(camera, torch.Tensor):
                camera = torch.as_tensor(camera)
            if camera.ndim != 4:
                raise ValueError(
                    f"{name}[{index}] must be a 4-D image batch; got "
                    f"shape={tuple(camera.shape)}."
                )
            if camera.shape[1] not in (1, 3, 4) and camera.shape[-1] in (1, 3, 4):
                camera = camera.permute(0, 3, 1, 2).contiguous()
            normalized.append(camera)
        return normalized

    def _build_request(self, env_obs: dict[str, Any]) -> tuple[PI05Request, Any]:
        if self._processor is None:
            raise RuntimeError("init_worker() must be called before PhyAI inference.")
        if self._engine_device is None or self._engine_dtype is None:
            raise RuntimeError("PhyAI engine device and dtype are not initialized.")

        images = self._camera_batches(env_obs.get("main_images"), "main_images")
        images.extend(self._camera_batches(env_obs.get("wrist_images"), "wrist_images"))
        images.extend(
            self._camera_batches(env_obs.get("extra_view_images"), "extra_view_images")
        )
        if self._normalize_pixels:
            images = [
                image.float().div(255.0) if image.dtype == torch.uint8 else image
                for image in images
            ]
        if not images:
            raise ValueError("PhyAI inference requires env_obs['main_images'].")
        processed = self._processor.preprocess(
            {
                "images": images,
                "task": env_obs["task_descriptions"],
                "state": env_obs["states"].to(device=self._engine_device),
            }
        )
        request = PI05Request(
            pixel_values=processed.pixel_values.to(
                device=self._engine_device, dtype=self._engine_dtype
            ),
            input_ids=processed.input_ids.to(device=self._engine_device),
            lang_lens=processed.lang_lens.to(device=self._engine_device),
        )
        return request, processed

    def _build_training_forward_inputs(
        self,
        env_obs: dict[str, Any],
        processed: Any,
        rollout_result: Any,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build the legacy native OpenPI replay contract.

        The Actor intentionally receives raw observations and runs its existing
        OpenPI transforms again. Only prompt tokens are reused from PhyAI so the
        behavior and replay policies see exactly the same task tokenization.
        """
        batch_size = int(rollout_result.actions.shape[0])
        token_ids = processed.input_ids.to(device=self._engine_device)
        token_mask = (
            torch.arange(token_ids.shape[1], device=self._engine_device)[None, :]
            < processed.lang_lens.to(device=self._engine_device)[:, None]
        )

        states = env_obs["states"]
        state_indices = self.model_cfg.get("openpi", {}).get("state_indices", None)
        if state_indices is not None:
            states = states[..., list(state_indices)]

        forward_inputs = {
            "chains": rollout_result.chains,
            "denoise_inds": rollout_result.denoise_inds,
            "tokenized_prompt": token_ids,
            "tokenized_prompt_mask": token_mask,
            "action": actions.to(device=self._engine_device).reshape(batch_size, -1),
            "model_action": rollout_result.actions.reshape(batch_size, -1),
            "observation/image": env_obs["main_images"],
            "observation/state": states,
        }
        if env_obs.get("wrist_images") is not None:
            forward_inputs["observation/wrist_image"] = env_obs["wrist_images"]
        if env_obs.get("extra_view_images") is not None:
            forward_inputs["observation/extra_view_image"] = env_obs[
                "extra_view_images"
            ]
        return {
            key: value.contiguous() if torch.is_tensor(value) else value
            for key, value in forward_inputs.items()
        }

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Run inference or collect the real behavior-policy PPO state."""
        if mode not in ("train", "eval"):
            raise ValueError(f"Unsupported PhyAI rollout mode: {mode!r}.")
        if mode == "train" and self.only_eval:
            raise RuntimeError("An evaluation-only PhyAI worker cannot run train mode.")
        if self._engine is None or self._processor is None:
            raise RuntimeError("init_worker() must be called before PhyAI inference.")

        request, processed = self._build_request(env_obs)
        rollout_result = None
        if mode == "train":
            openpi_cfg = self.model_cfg.get("openpi", {})
            rollout_config = PI05RolloutConfig(
                action_chunk=int(self.model_cfg.num_action_chunks),
                action_dim=int(self.model_cfg.action_dim),
                noise_method=str(openpi_cfg.get("noise_method", "flow_sde")),
                noise_level=float(openpi_cfg.get("noise_level", 0.5)),
            )
            rollout_result = self._engine.rollout_step(
                request, rollout_config=rollout_config
            )
            model_actions = rollout_result.actions
        else:
            model_actions = self._engine.step(request)
        actions = self._processor.postprocess(model_actions)
        if not isinstance(actions, torch.Tensor):
            actions = torch.as_tensor(actions)
        requested_chunks = int(self.model_cfg.num_action_chunks)
        if actions.shape[1] < requested_chunks:
            raise ValueError(
                "PhyAI returned fewer action chunks than RLinf requested: "
                f"returned={actions.shape[1]}, requested={requested_chunks}."
            )
        actions = actions[:, :requested_chunks]
        actions = actions.to(dtype=torch.float32).contiguous()

        batch_size = actions.shape[0]
        if rollout_result is None:
            result = {
                "prev_logprobs": None,
                "prev_values": None,
                "forward_inputs": {
                    "action": actions.reshape(batch_size, -1),
                    "model_action": model_actions.reshape(batch_size, -1),
                },
                "expert_label_flag": False,
            }
        else:
            result = {
                "prev_logprobs": rollout_result.prev_logprobs,
                "prev_values": rollout_result.prev_values,
                "forward_inputs": self._build_training_forward_inputs(
                    env_obs, processed, rollout_result, actions
                ),
                "expert_label_flag": False,
            }
        return actions, result

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        """Compute the final-observation value for native GAE."""
        if final_obs is None or self.only_eval:
            return None
        with torch.no_grad():
            _, result = self.predict(final_obs, mode="train")
        values = result["prev_values"]
        if values is None:
            raise RuntimeError(
                "PhyAI training rollout did not return bootstrap values."
            )
        return values[:, :1].cpu().contiguous()

    @Worker.timer("sync_model_from_actor")
    async def sync_model_from_actor(self) -> None:
        """Receive actor buckets and hot-update the in-process PhyAI engine."""
        if self._engine is None or self._weight_target is None:
            raise RuntimeError("init_worker() must be called before weight sync.")
        if self.weight_syncer is None:
            raise RuntimeError("PhyAI weight sync requires weight_syncer config.")
        if not isinstance(self.weight_syncer, BucketWeightSyncer):
            raise NotImplementedError(
                "PhyAI currently supports only bucket weight synchronization. "
                "Patch synchronization assumes identical sender/receiver state "
                "dict layouts, which is incompatible with PhyAI fused weights."
            )

        async def recv_func() -> Any:
            return await self.broadcast(
                None,
                groups=[
                    (self.actor_group_name, self.actor_weight_src_rank),
                    (self._group_name, self._weight_sync_rollout_ranks),
                ],
                src=(self.actor_group_name, self.actor_weight_src_rank),
                async_op=True,
                options=self._sync_weight_comm_options,
            ).async_wait()

        async def send_func(data: Any) -> None:
            if not self._weight_sync_is_sender:
                return
            actor_world_size = self.placement.get_world_size("actor")
            for actor_rank in range(actor_world_size):
                await self.send(
                    data,
                    dst_group_name=self.actor_group_name,
                    dst_rank=actor_rank,
                    async_op=True,
                    options=self._sync_weight_comm_options,
                ).async_wait()

        if not self.weight_syncer.receiver_initialized():
            await self.weight_syncer.init_receiver(
                state_dict=None,
                recv=recv_func,
                send=send_func,
            )

        self._engine.begin_weight_update()
        try:
            applied_version = await self.weight_syncer.apply(
                self._weight_target,
                recv_func,
            )
            report = self._engine.finish_weight_update(version=applied_version)
        except Exception:
            self._engine.abort_weight_update()
            raise

        self.version = applied_version
        if self.finished_episodes is None:
            self.finished_episodes = (
                self.version * self.total_num_train_envs * self.rollout_epoch
            )
        self.log_info(
            "PhyAI hot weight update applied: "
            f"version={self.version}, loaded={len(report.loaded)}."
        )
        gc.collect()
        self.torch_platform.empty_cache()

    def set_global_step(self, global_step: int) -> None:
        """Record the policy version without forwarding to an HF model."""
        self.version = int(global_step)

    def shutdown(self) -> None:
        """Release the PhyAI entry and its process-local distributed state."""
        if self._engine is None:
            return
        self.log_info(f"Shutting down PhyAI engine on rollout rank {self._rank}.")
        self._engine.close()
        self._engine = None
        self._weight_target = None
        self._processor = None
