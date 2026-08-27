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

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.hybrid_engines.weight_syncer.bucket_syncer import BucketWeightSyncer
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.workers.rollout.phyai import get_embodied_rollout_worker
from rlinf.workers.rollout.phyai.phyai_worker import (
    PhyAIWorker,
    _PhyAIWeightTarget,
)


class _FakeEngine:
    def __init__(self) -> None:
        self.events = []

    def begin_weight_update(self) -> None:
        self.events.append("begin")

    def update_weights(self, state_dict) -> None:
        self.events.append(("update", state_dict))

    def finish_weight_update(self, version=None):
        self.events.append(("finish", version))
        return SimpleNamespace(loaded=["model.weight"])

    def abort_weight_update(self) -> None:
        self.events.append("abort")


class _FakeBucketWeightSyncer(BucketWeightSyncer):
    def __init__(self, *, fail: bool = False) -> None:
        super().__init__(
            bucket_size=1024,
            bucket_dtype=None,
            bucket_device="cpu",
        )
        self.fail = fail

    async def init_receiver(self, state_dict, recv, send=None) -> None:
        del state_dict, recv, send
        self._receiver_initialized = True

    async def apply(self, model, recv) -> int:
        del recv
        model.load_state_dict({"model.weight": torch.ones(2, 2)})
        if self.fail:
            raise RuntimeError("sync failed")
        return 7


def _make_worker(*, fail: bool = False):
    worker = object.__new__(PhyAIWorker)
    engine = _FakeEngine()
    worker._accelerator_type = "cpu"
    worker._timer_metrics = {}
    worker._rank = 0
    worker._engine = engine
    worker._weight_target = _PhyAIWeightTarget(engine)
    worker.weight_syncer = _FakeBucketWeightSyncer(fail=fail)
    worker.actor_group_name = "actor"
    worker.actor_weight_src_rank = 0
    worker._group_name = "rollout"
    worker._weight_sync_rollout_ranks = [0]
    worker._weight_sync_is_sender = False
    worker._sync_weight_comm_options = None
    worker.finished_episodes = None
    worker.total_num_train_envs = 4
    worker.rollout_epoch = 2
    worker.version = 0
    worker.torch_platform = SimpleNamespace(empty_cache=lambda: None)
    worker.log_info = lambda _message: None
    return worker, engine


@pytest.mark.asyncio
async def test_sync_model_from_actor_streams_buckets_into_engine():
    worker, engine = _make_worker()

    await PhyAIWorker.sync_model_from_actor.__wrapped__(worker)

    assert engine.events[0] == "begin"
    assert engine.events[1][0] == "update"
    assert engine.events[2] == ("finish", 7)
    assert worker.version == 7
    assert worker.finished_episodes == 56


@pytest.mark.asyncio
async def test_sync_model_from_actor_aborts_failed_update():
    worker, engine = _make_worker(fail=True)

    with pytest.raises(RuntimeError, match="sync failed"):
        await PhyAIWorker.sync_model_from_actor.__wrapped__(worker)

    assert engine.events[-1] == "abort"


def test_embodied_rollout_backend_is_opt_in():
    default_cfg = OmegaConf.create({"rollout": {}})
    native_cfg = OmegaConf.create({"rollout": {"rollout_backend": "huggingface"}})
    phyai_cfg = OmegaConf.create({"rollout": {"rollout_backend": "phyai"}})

    assert get_embodied_rollout_worker(default_cfg) is MultiStepRolloutWorker
    assert get_embodied_rollout_worker(native_cfg) is MultiStepRolloutWorker
    assert get_embodied_rollout_worker(phyai_cfg) is PhyAIWorker


def test_phyai_weight_target_converts_openpi_rlinf_layout():
    engine = _FakeEngine()
    target = _PhyAIWeightTarget(engine, openpi_rlinf_layout=True)
    qkv = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    gating = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    value_bias = torch.ones(4)

    target.load_state_dict(
        {
            "model.img.encoder.layers.0.attn.in_proj_weight": qkv,
            "model.llm.layers.0.mlps.1.w_gating": gating,
            "value_head.mlp.0.bias": value_bias,
        }
    )

    weights = engine.events[0][1]
    vision_prefix = (
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model."
        "encoder.layers.0.self_attn"
    )
    assert torch.equal(weights[f"{vision_prefix}.q_proj.weight"], qkv[:2])
    assert torch.equal(weights[f"{vision_prefix}.k_proj.weight"], qkv[2:4])
    assert torch.equal(weights[f"{vision_prefix}.v_proj.weight"], qkv[4:])
    expert_prefix = "paligemma_with_expert.gemma_expert.model.layers.0.mlp"
    assert torch.equal(weights[f"{expert_prefix}.gate_proj.weight"], gating[0].T)
    assert torch.equal(weights[f"{expert_prefix}.up_proj.weight"], gating[1].T)
    assert weights["value_head.mlp.0.bias"] is value_bias


def test_phyai_weight_target_keeps_legacy_layout_unchanged():
    engine = _FakeEngine()
    target = _PhyAIWeightTarget(engine)
    weights = {"model.weight": torch.ones(2, 2)}

    target.load_state_dict(weights)

    assert engine.events == [("update", weights)]


@pytest.mark.parametrize("plugin", ["pi05", "pi05_rl"])
def test_phyai_worker_accepts_supported_plugins(monkeypatch, plugin):
    def _init_base(worker, _cfg):
        worker.enable_offload = False
        worker.global_accelerator_ids = [0]

    monkeypatch.setattr(MultiStepRolloutWorker, "__init__", _init_base)
    cfg = OmegaConf.create({"rollout": {"phyai": {"plugin": plugin}}})

    worker = PhyAIWorker(cfg)

    assert worker._phyai_plugin == plugin


def test_phyai_worker_rejects_unknown_plugin(monkeypatch):
    def _init_base(worker, _cfg):
        worker.enable_offload = False
        worker.global_accelerator_ids = [0]

    monkeypatch.setattr(MultiStepRolloutWorker, "__init__", _init_base)
    cfg = OmegaConf.create({"rollout": {"phyai": {"plugin": "unknown"}}})

    with pytest.raises(NotImplementedError, match="pi05_rl"):
        PhyAIWorker(cfg)


def test_training_forward_inputs_match_native_contract():
    worker = object.__new__(PhyAIWorker)
    worker._engine_device = torch.device("cpu")
    worker.model_cfg = OmegaConf.create({"openpi": {}})
    env_obs = {
        "main_images": torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.uint8),
        "wrist_images": torch.randint(0, 256, (2, 3, 128, 128), dtype=torch.uint8),
        "extra_view_images": None,
        "states": torch.randn(2, 8),
    }
    processed = SimpleNamespace(
        state=torch.randn(2, 8),
        input_ids=torch.arange(400).view(2, 200),
        lang_lens=torch.tensor([7, 8]),
        pixel_values=torch.randn(2, 2, 3, 224, 224),
    )
    rollout = SimpleNamespace(
        actions=torch.randn(2, 10, 32),
        chains=torch.randn(2, 4, 10, 32),
        denoise_inds=torch.ones(2, 3, dtype=torch.long),
    )
    actions = torch.randn(2, 5, 7)

    inputs = worker._build_training_forward_inputs(env_obs, processed, rollout, actions)

    assert inputs["chains"].shape == (2, 4, 10, 32)
    assert inputs["denoise_inds"].shape == (2, 3)
    assert inputs["tokenized_prompt_mask"].sum(dim=1).tolist() == [7, 8]
    assert inputs["observation/image"] is env_obs["main_images"]
    assert inputs["observation/wrist_image"] is env_obs["wrist_images"]
    assert inputs["observation/state"] is env_obs["states"]
    assert "obs_state" not in inputs
    assert not any(key.startswith("obs_image__") for key in inputs)
    assert inputs["action"].shape == (2, 35)
    assert inputs["model_action"].shape == (2, 320)


def test_training_forward_inputs_match_openpi_rlinf_contract():
    worker = object.__new__(PhyAIWorker)
    worker._engine_device = torch.device("cpu")
    worker.model_cfg = OmegaConf.create(
        {
            "model_type": "openpi_rlinf",
            "openpi": {"model_action_dim": 32},
        }
    )
    env_obs = {"states": torch.randn(2, 8)}
    processed = SimpleNamespace(
        state=torch.randn(2, 8),
        input_ids=torch.arange(400).view(2, 200),
        lang_lens=torch.tensor([7, 8]),
        pixel_values=torch.randn(2, 2, 3, 224, 224),
    )
    rollout = SimpleNamespace(
        actions=torch.randn(2, 10, 32),
        chains=torch.randn(2, 4, 10, 32),
        denoise_inds=torch.ones(2, 3, dtype=torch.long),
    )

    inputs = worker._build_training_forward_inputs(
        env_obs, processed, rollout, torch.randn(2, 5, 7)
    )

    assert inputs["obs_state"].shape == (2, 32)
    assert inputs["tokenized_prompt_mask"].sum(dim=1).tolist() == [7, 8]
    assert inputs["obs_image__base_0_rgb"].shape == (2, 224, 224, 3)
    assert inputs["obs_image__left_wrist_0_rgb"].shape == (2, 224, 224, 3)
    assert not inputs["obs_image__right_wrist_0_rgb"].any()
    assert inputs["obs_image_mask__base_0_rgb"].all()
    assert inputs["obs_image_mask__left_wrist_0_rgb"].all()
    assert not inputs["obs_image_mask__right_wrist_0_rgb"].any()
    assert "observation/state" not in inputs
