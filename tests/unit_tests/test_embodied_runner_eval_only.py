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

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rlinf.runners.embodied_runner import EmbodiedRunner


def test_run_only_eval_syncs_restored_actor_before_evaluation():
    runner = EmbodiedRunner.__new__(EmbodiedRunner)
    runner.cfg = SimpleNamespace(runner={"only_eval": True})
    runner.global_step = 500
    runner.max_steps = 500
    runner.actor = MagicMock()
    runner.rollout = MagicMock()
    runner.timer = MagicMock()
    runner.timer.return_value.__enter__.return_value = None
    runner.metric_logger = MagicMock(log_path="/tmp/eval")
    runner.update_rollout_weights = MagicMock()
    runner.evaluate = MagicMock(return_value={"success_rate": 0.996})
    runner._finish_run = MagicMock()

    with patch("rlinf.runners.embodied_runner.print_metrics_table") as print_table:
        result = runner.run()

    runner.actor.set_global_step.assert_called_once_with(500)
    runner.actor.set_global_step.return_value.wait.assert_called_once_with()
    runner.rollout.set_global_step.assert_called_once_with(500)
    runner.rollout.set_global_step.return_value.wait.assert_called_once_with()
    runner.update_rollout_weights.assert_called_once_with()
    runner.evaluate.assert_called_once_with()
    runner.metric_logger.log.assert_called_once_with(
        data={"eval/success_rate": 0.996}, step=500
    )
    print_table.assert_called_once()
    runner._finish_run.assert_called_once_with()
    assert result == {"eval/success_rate": 0.996}
