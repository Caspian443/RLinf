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
from typing import Any, Literal

import torch
from omegaconf import DictConfig, OmegaConf

from rlinf.config import torch_dtype_from_precision
from rlinf.scheduler import Worker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class _PhyAIWeightTarget(torch.nn.Module):
    """Adapt bucket syncer's ``load_state_dict`` calls to a PhyAI Engine."""

    def __init__(self, engine: Any) -> None:
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

    The current PhyAI pi0.5 engine returns actions only. Consequently this
    first integration supports evaluation and hot weight synchronization, but
    deliberately rejects training rollout until PhyAI returns RLinf's
    behavior-policy tensors.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

        self._phyai_cfg = cfg.rollout.get("phyai", {})
        self._phyai_plugin = str(self._phyai_cfg.get("plugin", "pi05")).lower()
        self._engine = None
        self._processor = None
        self._request_cls = None
        self._weight_target = None
        self._engine_device: torch.device | None = None
        self._engine_dtype: torch.dtype | None = None

        if self._phyai_plugin != "pi05":
            raise NotImplementedError(
                "PhyAIWorker currently supports only the 'pi05' engine plugin; "
                f"got {self._phyai_plugin!r}."
            )
        if not self.only_eval:
            raise NotImplementedError(
                "PhyAI training rollout is not implemented yet: Engine.step() "
                "must first expose prev_logprobs, prev_values, denoise state, "
                "and forward_inputs. "
                "Set runner.only_eval: true for the initial integration."
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

        from phyai.engine import Engine, EngineArgs
        from phyai.engine_config import DeviceConfig, EngineConfig, RuntimeConfig
        from phyai.models.pi05.configuration_pi05 import PI05Config
        from phyai.models.pi05.main_pi05 import PI05Args
        from phyai.models.pi05.scheduler_ws1_pi05 import PI05Request
        from phyai.utils import load_config
        from phyai_utils_tools.models.pi05 import PI05Processor

        checkpoint_dir = str(
            self._phyai_cfg.get("checkpoint_dir", None) or self.model_cfg.model_path
        )
        plugin_cfg = load_config(checkpoint_dir, PI05Config)

        dtype = torch_dtype_from_precision(self.model_cfg.precision)
        if dtype is None:
            raise ValueError(
                f"Unsupported PhyAI model precision: {self.model_cfg.precision!r}."
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
        runtime_values.setdefault(
            "use_cuda_graph", self._phyai_cfg.get("use_cuda_graph", True)
        )

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
                    vision_params_dtype=vision_dtype,
                    inputs_image_shape=inputs_image_shape,
                ),
                config=EngineConfig(
                    device=DeviceConfig(
                        target=str(self._engine_device), params_dtype=dtype
                    ),
                    runtime=RuntimeConfig(**runtime_values),
                ),
            )
        )

        processor_kwargs = {
            "image_size": plugin_cfg.vision.image_size,
            "num_channels": plugin_cfg.vision.num_channels,
            "num_images": num_images,
            "action_dim": int(self.model_cfg.action_dim),
            "normalize_pixels": bool(self._phyai_cfg.get("normalize_pixels", False)),
            "image_pad_value": float(self._phyai_cfg.get("image_pad_value", 0.0)),
            "device": self._engine_device,
            "params_dtype": dtype,
        }
        try:
            if self._phyai_cfg.get("processor_from_pretrained", True):
                processor = PI05Processor.from_pretrained(
                    checkpoint_dir, **processor_kwargs
                )
            else:
                processor = PI05Processor(
                    tokenizer_max_length=plugin_cfg.tokenizer_max_length,
                    **processor_kwargs,
                )
        except Exception:
            engine.close()
            raise

        self._engine = engine
        self._weight_target = _PhyAIWeightTarget(engine)
        self._processor = processor
        self._request_cls = PI05Request

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

    def _build_request(self, env_obs: dict[str, Any]):
        if self._processor is None or self._request_cls is None:
            raise RuntimeError("init_worker() must be called before PhyAI inference.")
        if self._engine_device is None or self._engine_dtype is None:
            raise RuntimeError("PhyAI engine device and dtype are not initialized.")

        images = self._camera_batches(env_obs.get("main_images"), "main_images")
        images.extend(self._camera_batches(env_obs.get("wrist_images"), "wrist_images"))
        images.extend(
            self._camera_batches(env_obs.get("extra_view_images"), "extra_view_images")
        )
        if not images:
            raise ValueError("PhyAI inference requires env_obs['main_images'].")
        processed = self._processor.preprocess(
            {
                "images": images,
                "task": env_obs["task_descriptions"],
                "state": env_obs["states"],
            }
        )
        return self._request_cls(
            pixel_values=processed.pixel_values.to(
                device=self._engine_device, dtype=self._engine_dtype
            ),
            input_ids=processed.input_ids.to(device=self._engine_device),
            lang_lens=processed.lang_lens.to(device=self._engine_device),
        )

    @Worker.timer("predict")
    def predict(
        self, env_obs: dict[str, Any], mode: Literal["train", "eval"] = "eval"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Convert RLinf observations, run ``Engine.step``, and return actions."""
        if mode != "eval":
            raise NotImplementedError(
                "PhyAI Engine.step() currently returns actions only and cannot "
                "satisfy RLinf's training rollout result contract."
            )
        if self._engine is None or self._processor is None:
            raise RuntimeError("init_worker() must be called before PhyAI inference.")

        request = self._build_request(env_obs)
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
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(batch_size, -1),
                "model_action": model_actions.reshape(batch_size, -1),
            },
            "expert_label_flag": False,
        }
        return actions, result

    def get_bootstrap_values(
        self, final_obs: dict[str, Any] | None
    ) -> torch.Tensor | None:
        """PhyAI pi0.5 currently has no value head."""
        return None

    @Worker.timer("sync_model_from_actor")
    async def sync_model_from_actor(self) -> None:
        """Receive actor buckets and hot-update the in-process PhyAI engine."""
        from rlinf.hybrid_engines.weight_syncer.bucket_syncer import (
            BucketWeightSyncer,
        )

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
            report = self._engine.finish_weight_update()
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
        self._request_cls = None
