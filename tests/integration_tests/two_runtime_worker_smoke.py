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

"""Launch two RLinf WorkerGroups with different Python interpreters."""

import argparse
import json
import os
import sys

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker


class RuntimeProbeWorker(Worker):
    """Report the runtime that Ray actually selected for this worker."""

    def probe(self, role: str) -> dict[str, object]:
        import torch
        import transformers

        result: dict[str, object] = {
            "role": role,
            "pid": os.getpid(),
            "python": sys.executable,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
        if role == "actor":
            import openpi
            from openpi.models_pytorch import pi0_pytorch

            result.update(
                openpi=openpi.__file__,
                pi0_pytorch=pi0_pytorch.__file__,
            )
        elif role == "rollout":
            import phyai
            from phyai.engine import Engine

            result.update(
                phyai=phyai.__file__,
                engine=f"{Engine.__module__}.{Engine.__qualname__}",
            )
        else:
            raise ValueError(f"Unknown role: {role}")
        return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-python", required=True)
    parser.add_argument("--rollout-python", required=True)
    args = parser.parse_args()

    cluster = Cluster(num_nodes=1)
    placement = NodePlacementStrategy([0])
    actor = RuntimeProbeWorker.create_group().launch(
        cluster,
        placement_strategy=placement,
        name="TwoRuntimeActorProbe",
        python_interpreter_path=args.actor_python,
    )
    rollout = RuntimeProbeWorker.create_group().launch(
        cluster,
        placement_strategy=placement,
        name="TwoRuntimeRolloutProbe",
        python_interpreter_path=args.rollout_python,
    )

    evidence = {
        "actor": actor.probe("actor").wait()[0],
        "rollout": rollout.probe("rollout").wait()[0],
        "driver": {"pid": os.getpid(), "python": sys.executable},
    }
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
