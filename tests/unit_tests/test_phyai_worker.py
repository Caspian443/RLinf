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

from rlinf.hybrid_engines.weight_syncer.bucket_syncer import BucketWeightSyncer
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

    def finish_weight_update(self):
        self.events.append("finish")
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
    assert engine.events[2] == "finish"
    assert worker.version == 7
    assert worker.finished_episodes == 56


@pytest.mark.asyncio
async def test_sync_model_from_actor_aborts_failed_update():
    worker, engine = _make_worker(fail=True)

    with pytest.raises(RuntimeError, match="sync failed"):
        await PhyAIWorker.sync_model_from_actor.__wrapped__(worker)

    assert engine.events[-1] == "abort"
