# Copyright 2025 The RLinf Authors.
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

from omegaconf import DictConfig

from rlinf.scheduler.worker.worker import Worker


def get_actor_worker(cfg: DictConfig) -> Worker:
    if cfg.actor.training_backend == "fsdp":
        task_type = cfg.runner.get("task_type", "")
        model_type = cfg.actor.model.get("model_type", "")
        generation_backend = cfg.rollout.get("generation_backend", "")
        
        # Use AVFSDPActor for AV tasks
        if task_type == "av" or generation_backend == "av":
            from .fsdp_actor_worker import AVFSDPActor
            return AVFSDPActor
        
        # Use EmbodiedFSDPActor for embodied tasks (needs env worker)
        if (
            task_type == "embodied" or
            model_type in ["openvla", "openvla_oft", "gr00t"] or
            "env" in cfg.get("cluster", {}).get("component_placement", {})
        ):
            from .fsdp_actor_worker import EmbodiedFSDPActor
            return EmbodiedFSDPActor
        
        # Use standard FSDPActor for LLM tasks
        from .fsdp_actor_worker import FSDPActor
        return FSDPActor
        
    elif cfg.actor.training_backend == "megatron":
        from .megatron_actor_worker import MegatronActor

        return MegatronActor
    else:
        raise ValueError(f"Unsupported training backend: {cfg.actor.training_backend}")
