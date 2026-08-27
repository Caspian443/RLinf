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

import pytest
from omegaconf import OmegaConf

from rlinf.models.embodiment.openpi_rlinf import _resolve_model_action_horizon


def test_model_action_horizon_defaults_to_executed_chunk():
    cfg = OmegaConf.create(
        {
            "num_action_chunks": 5,
            "openpi": {},
        }
    )

    assert _resolve_model_action_horizon(cfg, cfg.openpi) == 5


def test_model_action_horizon_can_exceed_executed_chunk():
    cfg = OmegaConf.create(
        {
            "num_action_chunks": 5,
            "openpi": {"model_action_horizon": 10},
        }
    )

    assert _resolve_model_action_horizon(cfg, cfg.openpi) == 10


def test_model_action_horizon_rejects_shorter_model_output():
    cfg = OmegaConf.create(
        {
            "num_action_chunks": 5,
            "openpi": {"model_action_horizon": 4},
        }
    )

    with pytest.raises(ValueError, match="must be at least"):
        _resolve_model_action_horizon(cfg, cfg.openpi)
