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

import inspect
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("phyai")

from phyai.models.pi05.scheduler_pi05 import PI05RolloutRequest

from rlinf.hybrid_engines.weight_syncer.bucket_syncer import BucketWeightSyncer
from rlinf.models.embodiment.openpi.modules.model import (
    IMAGE_KEYS,
    Observation,
)
from rlinf.workers.rollout import utils as rollout_utils
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker
from rlinf.workers.rollout.phyai import phyai_worker
from rlinf.workers.rollout.phyai.phyai_worker import PhyAIWorker, _PhyAIWeightTarget


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


def _make_rollout_worker(*, fail: bool = False):
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
async def test_sync_model_from_actor_commits_only_complete_updates():
    worker, engine = _make_rollout_worker()
    await inspect.unwrap(PhyAIWorker.sync_model_from_actor)(worker)

    event_names = [
        event[0] if isinstance(event, tuple) else event for event in engine.events
    ]
    assert event_names == ["begin", "update", "finish"]
    assert engine.events[-1] == ("finish", 7)
    assert worker.version == 7
    assert worker.finished_episodes == 56

    worker, engine = _make_rollout_worker(fail=True)
    with pytest.raises(RuntimeError, match="sync failed"):
        await inspect.unwrap(PhyAIWorker.sync_model_from_actor)(worker)
    assert engine.events[-1] == "abort"
    assert worker.version == 0


def test_embodied_rollout_backend_selection():
    hf_cfg = OmegaConf.create({"rollout": {"rollout_backend": "hf"}})
    phyai_cfg = OmegaConf.create({"rollout": {"rollout_backend": "phyai"}})

    assert rollout_utils.get_embodied_rollout_worker(hf_cfg) is MultiStepRolloutWorker
    assert rollout_utils.get_embodied_rollout_worker(phyai_cfg) is PhyAIWorker


def test_phyai_weight_target_converts_actor_layout():
    engine = _FakeEngine()
    target = _PhyAIWeightTarget(engine)
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


def test_phyai_worker_rejects_unknown_plugin(monkeypatch):
    def _init_base(worker, _cfg):
        worker.enable_offload = False
        worker.global_accelerator_ids = [0]

    monkeypatch.setattr(MultiStepRolloutWorker, "__init__", _init_base)
    cfg = OmegaConf.create({"rollout": {"phyai": {"plugin": "unknown"}}})

    with pytest.raises(NotImplementedError, match="pi05"):
        PhyAIWorker(cfg)


@pytest.mark.parametrize("only_eval", [False, True])
def test_init_worker_loads_checkpoint_and_builds_phyai_processors(
    monkeypatch, only_eval
):
    captured = {}

    class _InitializedEngine:
        def __init__(self, args):
            captured["engine_args"] = args

    monkeypatch.setattr(phyai_worker, "KernelConfig", SimpleNamespace)
    monkeypatch.setattr(phyai_worker, "RuntimeConfig", SimpleNamespace)
    monkeypatch.setattr(phyai_worker, "DeviceConfig", SimpleNamespace)
    monkeypatch.setattr(phyai_worker, "EngineConfig", SimpleNamespace)
    monkeypatch.setattr(phyai_worker, "PI05Args", SimpleNamespace)
    monkeypatch.setattr(phyai_worker, "EngineArgs", SimpleNamespace)
    monkeypatch.setattr(phyai_worker, "Engine", _InitializedEngine)

    processor = SimpleNamespace()
    preprocessor = SimpleNamespace()
    monkeypatch.setattr(
        PhyAIWorker,
        "_build_phyai_processors",
        lambda *_args, **_kwargs: (processor, preprocessor),
    )

    worker = object.__new__(PhyAIWorker)
    worker._engine = None
    worker._processor = None
    worker._phyai_preprocessor = None
    worker._phyai_plugin = "pi05"
    worker._phyai_cfg = OmegaConf.create({"params_dtype": "bf16"})
    worker.model_cfg = OmegaConf.create(
        {
            "model_path": "/checkpoint",
            "model_type": "openpi",
            "precision": "bf16",
            "num_action_chunks": 5,
            "action_dim": 7,
            "num_steps": 3,
            "add_value_head": True,
            "openpi": {
                "config_name": "pi05_libero",
                "action_horizon": 10,
                "model_action_dim": 32,
                "max_token_len": 200,
                "paligemma_variant": "gemma_2b",
                "action_expert_variant": "gemma_300m",
                "num_images_in_input": 2,
                "detach_critic_input": True,
                "value_after_vlm": True,
            },
        }
    )
    worker.only_eval = only_eval
    worker.torch_device_type = "cpu"
    worker.per_node_train_batch_size = 2
    worker.per_node_eval_batch_size = 2
    worker.log_info = lambda _message: None
    worker._rank = 0

    worker.init_worker()

    engine_args = captured["engine_args"]
    plugin_args = engine_args.plugin_args
    plugin_cfg = plugin_args.config
    assert engine_args.config.device.params_dtype is torch.bfloat16
    assert plugin_args.checkpoint_dir == "/checkpoint"
    assert plugin_cfg.chunk_size == 10
    assert plugin_cfg.max_action_dim == 32
    assert plugin_cfg.num_inference_steps == 3
    assert plugin_cfg.critic_action_chunk is None
    assert plugin_cfg.add_value_head
    assert plugin_cfg.value_after_vlm
    assert worker._processor is processor
    assert worker._phyai_preprocessor is preprocessor


def test_phyai_preprocessing_builds_task_only_rollout_and_replay_inputs():
    class _Processor:
        def postprocess(self, actions):
            return actions + 3

    class _Preprocessor:
        def __call__(self, raw):
            assert len(raw["images"]) == 2
            assert raw["prompt"] == ["pick cube\n", "place block\n"]
            batch_size = raw["images"][0].shape[0]
            return {
                "pixel_values": torch.full(
                    (batch_size, 2, 3, 224, 224),
                    0.25,
                    dtype=torch.bfloat16,
                ),
                "input_ids": torch.arange(8).expand(batch_size, -1),
                "lang_lens": torch.full((batch_size,), 4, dtype=torch.int64),
                "state": raw["state"].to(torch.bfloat16),
            }

    env_obs = {
        "main_images": torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8),
        "wrist_images": torch.randint(0, 256, (2, 224, 224, 3), dtype=torch.uint8),
        "states": torch.randn(2, 8),
        "task_descriptions": ["pick_cube", "place\nblock"],
    }
    worker = object.__new__(PhyAIWorker)
    worker._engine_device = torch.device("cpu")
    worker._engine_dtype = torch.bfloat16
    worker._processor = _Processor()
    worker._phyai_preprocessor = _Preprocessor()
    worker._phyai_cfg = OmegaConf.create({"normalize_pixels": True})
    worker._num_images = 2
    worker._state_indices = None

    request, replay = worker._build_request(env_obs)

    assert request.pixel_values.dtype is torch.bfloat16
    assert request.pixel_values.eq(0.25).all()
    assert torch.equal(request.input_ids, torch.arange(8).expand(2, -1))
    assert request.lang_lens.eq(4).all()
    assert replay.state.dtype is torch.float32
    assert torch.equal(replay.state, env_obs["states"].to(torch.bfloat16).float())
    assert replay.images[IMAGE_KEYS[0]].dtype is torch.bfloat16
    assert replay.images[IMAGE_KEYS[0]].eq(0.25).all()
    assert replay.images[IMAGE_KEYS[2]].eq(0).all()
    assert torch.equal(replay.tokenized_prompt, request.input_ids)
    assert worker._processor.postprocess(torch.zeros(2, 10, 32)).eq(3).all()


def test_training_predict_returns_actor_replay_contract(monkeypatch):
    batch_size = 2
    processed = Observation(
        images={key: torch.zeros(batch_size, 224, 224, 3) for key in IMAGE_KEYS},
        image_masks={
            IMAGE_KEYS[0]: torch.ones(batch_size, dtype=torch.bool),
            IMAGE_KEYS[1]: torch.ones(batch_size, dtype=torch.bool),
            IMAGE_KEYS[2]: torch.zeros(batch_size, dtype=torch.bool),
        },
        state=torch.randn(batch_size, 8),
        tokenized_prompt=torch.arange(400).view(batch_size, 200),
        tokenized_prompt_mask=torch.arange(200)[None] < torch.tensor([[7], [8]]),
    )

    transition_logprobs = torch.arange(
        batch_size * 3 * 10 * 32, dtype=torch.float32
    ).view(batch_size, 3, 10, 32)

    class _Engine:
        def rollout_step(self, request):
            self.request = request
            return SimpleNamespace(
                actions=torch.zeros(batch_size, 10, 32),
                chains=torch.zeros(batch_size, 4, 10, 32),
                transition_logprobs=transition_logprobs,
                raw_values=torch.tensor([[1.0], [2.0]]),
            )

    worker = object.__new__(PhyAIWorker)
    worker._accelerator_type = "cpu"
    worker._timer_metrics = {}
    worker._engine = _Engine()
    worker._processor = SimpleNamespace(postprocess=lambda actions: actions[:, :5, :7])
    worker._phyai_preprocessor = SimpleNamespace()
    worker._build_request = lambda _env_obs: (
        SimpleNamespace(
            pixel_values=torch.zeros(batch_size, 2, 3, 224, 224),
            input_ids=processed.tokenized_prompt,
            lang_lens=processed.tokenized_prompt_mask.sum(dim=-1),
        ),
        processed,
    )
    worker._engine_device = torch.device("cpu")
    worker._num_images = 2
    worker._model_action_horizon = 10
    worker._model_action_dim = 32
    worker._num_steps = 3
    worker.only_eval = False
    worker.model_cfg = OmegaConf.create(
        {
            "num_action_chunks": 5,
            "action_dim": 7,
            "add_value_head": True,
            "openpi": {
                "model_action_dim": 32,
                "noise_method": "flow_sde",
                "noise_level": 0.5,
                "joint_logprob": False,
                "ignore_last": True,
            },
        }
    )
    monkeypatch.setattr(phyai_worker.random, "randint", lambda _low, _high: 1)

    actions, result = inspect.unwrap(PhyAIWorker.predict)(worker, {}, mode="train")

    request = worker._engine.request
    assert isinstance(request, PI05RolloutRequest)
    assert request.pixel_values.dtype is torch.float32
    assert request.noise is None
    assert request.step_noise is None
    assert request.sigmas[:, 1].gt(0).all()
    assert actions.shape == (batch_size, 5, 7)
    assert torch.equal(result["prev_logprobs"], transition_logprobs[:, 1, :5, :7])
    assert torch.equal(result["prev_values"], torch.tensor([[1.0], [2.0]]))
    assert result["forward_inputs"]["chains"].shape == (batch_size, 4, 10, 32)
    assert result["forward_inputs"]["denoise_inds"].eq(1).all()


def test_rollout_plan_uses_actor_sampling_defaults():
    worker = object.__new__(PhyAIWorker)
    worker._engine_device = torch.device("cpu")
    worker._model_action_horizon = 10
    worker._model_action_dim = 32
    worker._num_steps = 3
    worker.model_cfg = OmegaConf.create(
        {
            "num_action_chunks": 5,
            "action_dim": 7,
            "add_value_head": False,
            "openpi": {},
        }
    )
    request = SimpleNamespace(
        pixel_values=torch.empty(2, 2, 3, 4, 4),
        input_ids=torch.zeros(2, 200, dtype=torch.int64),
        lang_lens=torch.ones(2, dtype=torch.int64),
    )

    plan = worker._build_rollout_plan(request)

    assert plan.request.sigmas.eq(0).all()


def test_joint_initial_logprob_uses_realized_chain():
    request = PI05RolloutRequest(
        pixel_values=torch.empty(1, 2, 3, 4, 4),
        input_ids=torch.empty(1, 2, dtype=torch.int64),
        lang_lens=torch.ones(1, dtype=torch.int64),
        sigmas=torch.zeros(1, 2),
    )
    plan = phyai_worker._PI05SamplingPlan(
        request=request,
        denoise_inds=torch.tensor([[0, 1]]),
        action_chunk=1,
        action_dim=1,
        joint_logprob=True,
    )
    rollout_result = SimpleNamespace(
        chains=torch.tensor([[[[2.0]], [[0.0]], [[0.0]]]]),
        transition_logprobs=torch.zeros(1, 2, 1, 1),
        raw_values=None,
    )

    actual, values = PhyAIWorker._reduce_rollout_outputs(plan, rollout_result)

    initial = PhyAIWorker._initial_logprob(torch.tensor([[[2.0]]]))
    assert torch.equal(actual, initial / 3)
    assert values is None
