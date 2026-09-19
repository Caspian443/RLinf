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

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.config import validate_embodied_cfg
from rlinf.models.embodiment.openpi_rlinf import (
    _resolve_model_action_horizon,
    transforms_pipeline,
)


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


def test_transform_pipeline_uses_model_config_as_single_source(monkeypatch):
    captured = {}

    def _build(model_path, config_name, data_kwargs=None):
        captured.update(
            model_path=model_path,
            config_name=config_name,
            data_kwargs=data_kwargs,
        )
        return ["input"], ["output"]

    monkeypatch.setattr(transforms_pipeline, "build_openpi_transforms", _build)
    model_cfg = OmegaConf.create(
        {
            "model_path": "/checkpoint",
            "openpi": {"config_name": "pi05_libero"},
            "openpi_data": {"norm_stats_path": "/assets/norm_stats.json"},
        }
    )

    result = transforms_pipeline.build_openpi_transforms_from_model_config(model_cfg)

    assert result == ("pi05_libero", ["input"], ["output"])
    assert captured == {
        "model_path": "/checkpoint",
        "config_name": "pi05_libero",
        "data_kwargs": {"norm_stats_path": "/assets/norm_stats.json"},
    }


def test_phyai_eval_and_actor_resolve_same_openpi_config(monkeypatch):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="libero_spatial_ppo_openpi_pi05_phyai")

    assert cfg.rollout.model.model_type == cfg.actor.model.model_type
    assert cfg.rollout.model.openpi.config_name == cfg.actor.model.openpi.config_name
    assert cfg.rollout.model.openpi.num_images_in_input == 2
    assert cfg.rollout.model.num_action_chunks == cfg.actor.model.num_action_chunks


def test_phyai_training_requires_bucket_weight_sync(monkeypatch):
    config_dir = Path(__file__).resolve().parents[2] / "examples/embodiment/config"
    monkeypatch.setenv("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="libero_spatial_ppo_openpi_pi05_phyai")
    cfg.weight_syncer.type = "patch"

    with pytest.raises(AssertionError, match="weight_syncer.type='bucket'"):
        validate_embodied_cfg(cfg)
