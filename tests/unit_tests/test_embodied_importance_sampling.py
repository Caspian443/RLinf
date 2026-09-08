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
import math
from contextlib import nullcontext

import pytest
import torch
from omegaconf import OmegaConf

import rlinf.algorithms  # noqa: F401
from rlinf.algorithms.registry import policy_loss
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class _ReplayModel:
    def __init__(self) -> None:
        self.training = True
        self.batch_sizes = []

    def eval(self) -> None:
        self.training = False

    def __call__(self, *, forward_inputs, **kwargs):
        del kwargs
        expected = forward_inputs["expected_logprobs"]
        self.batch_sizes.append(expected.shape[0])
        return {"logprobs": expected + 0.25}


def test_embodied_actor_recomputes_logprobs_in_micro_batches():
    worker = object.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create(
        {
            "actor": {
                "micro_batch_size": 4,
                "model": {"model_type": "openpi"},
            },
            "algorithm": {"logprob_forward_micro_batch_size": 4},
            "rollout": {},
        }
    )
    worker.device = torch.device("cpu")
    worker.amp_context = nullcontext()
    worker.is_weight_offloaded = False
    worker.model = _ReplayModel()
    expected = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
    worker.rollout_batch = {
        "prev_logprobs": torch.zeros_like(expected),
        "forward_inputs": {"expected_logprobs": expected},
    }

    inspect.unwrap(EmbodiedFSDPActor.recompute_logprobs)(worker)

    assert not worker.model.training
    assert worker.model.batch_sizes == [3, 3]
    assert torch.equal(worker.rollout_batch["recomputed_logprobs"], expected + 0.25)


def test_embodied_importance_sampling_uses_loss_logprob_granularity():
    rollout_logprobs = torch.zeros(2, 1, 2, dtype=torch.float32)
    recomputed_logprobs = torch.full_like(rollout_logprobs, math.log(2.0) / 2.0)

    loss, metrics = policy_loss(
        task_type="embodied",
        loss_type="actor",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=2,
        logprobs=recomputed_logprobs.clone(),
        old_logprobs=recomputed_logprobs,
        rollout_logprobs=rollout_logprobs,
        advantages=torch.ones(2, dtype=torch.float32),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        importance_sampling_clip=1.5,
    )

    assert loss.item() == pytest.approx(-1.5)
    assert metrics["actor/ratio"] == pytest.approx(1.0)
    assert metrics["actor/importance_sampling_weight"] == pytest.approx(2.0)
    assert metrics["actor/importance_sampling_weight_max"] == pytest.approx(2.0)
    assert metrics["actor/importance_sampling_clip_fraction"] == pytest.approx(1.0)
    assert metrics["actor/recomputed_logprob_abs_diff"] == pytest.approx(math.log(2.0))


def test_embodied_importance_sampling_metrics_broadcast_loss_mask():
    rollout_logprobs = torch.zeros(1, 2, 2, dtype=torch.float32)
    recomputed_logprobs = torch.full_like(rollout_logprobs, math.log(2.0))

    _, metrics = policy_loss(
        task_type="embodied",
        loss_type="actor",
        logprob_type="token_level",
        reward_type="action_level",
        single_action_dim=2,
        logprobs=recomputed_logprobs.clone(),
        old_logprobs=recomputed_logprobs,
        rollout_logprobs=rollout_logprobs,
        advantages=torch.ones(1, 2, dtype=torch.float32),
        loss_mask=torch.tensor([[True, False]]),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        importance_sampling_clip=2.0,
    )

    assert metrics["actor/importance_sampling_weight"] == pytest.approx(2.0)
    assert metrics["actor/importance_sampling_weight_max"] == pytest.approx(2.0)
