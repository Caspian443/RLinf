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

from typing import Any

from .phyai_worker import PhyAIWorker


def get_embodied_rollout_worker(cfg: Any):
    """Select the embodied rollout worker; PhyAI is explicit opt-in."""
    backend = str(cfg.rollout.get("rollout_backend", "huggingface")).lower()
    if backend == "phyai":
        return PhyAIWorker
    if backend in ("huggingface", "hf"):
        from rlinf.workers.rollout.hf.huggingface_worker import (
            MultiStepRolloutWorker,
        )

        return MultiStepRolloutWorker
    raise ValueError(
        f"Unsupported embodied rollout_backend={backend!r}; "
        "expected 'huggingface' or 'phyai'."
    )


PhyAIMultiStepRolloutWorker = PhyAIWorker

__all__ = [
    "PhyAIMultiStepRolloutWorker",
    "PhyAIWorker",
    "get_embodied_rollout_worker",
]
