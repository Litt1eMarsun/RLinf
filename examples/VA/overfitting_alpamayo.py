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

"""
Training script for Alpamayo-R1 with GRPO on AV trajectory prediction.
Uses ReasoningRunner + AVWorker + AlpamayoAVDataset.
"""

import json
import os
import sys
import time
from contextlib import contextmanager

# IMPORTANT: Set HuggingFace cache paths BEFORE importing any HF libraries
# This ensures physical_ai_av can find the cached dataset
os.environ['HF_HOME'] = '/data/workspace/RLinf/data/'
os.environ['HF_HUB_CACHE'] = '/data/workspace/RLinf/data/hub'

@contextmanager
def timer(name):
    """简单的计时上下文管理器"""
    t0 = time.time()
    print(f"⏱️  开始: {name}")
    yield
    print(f"✓ 完成: {name} [{time.time()-t0:.1f}秒]")

# # Fix Ray initialization issue with psutil/uv detection in restricted environments
# os.environ['RAY_RUNTIME_ENV_SKIP_UV_CHECK'] = '1'
# os.environ['RAY_DISABLE_IMPORT_WARNING'] = '1'
# os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'

# Monkey patch psutil if /proc is not available
# try:
#     import psutil
#     # Test if psutil works
#     psutil.pids()
# except Exception as e:
#     print(f"Warning: psutil not working properly ({e}), applying workaround...")
#     # Create a minimal mock for psutil.Process if needed
#     import psutil as psutil_module
#     original_process = psutil_module.Process
    
#     class SafeProcess:
#         def __init__(self, pid=None):
#             try:
#                 self._process = original_process(pid)
#             except:
#                 self._process = None
        
#         def parents(self):
#             if self._process:
#                 try:
#                     return self._process.parents()
#                 except:
#                     return []
#             return []
        
#         def __getattr__(self, name):
#             if self._process:
#                 return getattr(self._process, name)
#             return lambda: None
    
#     psutil_module.Process = SafeProcess

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.data.datasets.alpamayo_av import AlpamayoAVDataset
from rlinf.runners.av_runner import AVRunner
from rlinf.scheduler import Cluster, NodePlacementStrategy
from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.utils.utils import output_redirector
from rlinf.workers.actor import get_actor_worker
from rlinf.workers.rollout.hf.av_worker import AVWorker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="alpamayo_r1_grpo_overfitting")
@output_redirector
def main(cfg) -> None:
    """Main training function."""
    startup_time = time.time()
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    # Create cluster and component placement
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ModelParallelComponentPlacement(cfg, cluster)

    # Create datasets
    with timer("加载训练数据集"):
        train_ds = AlpamayoAVDataset(cfg.data.train)
        print(f"  训练样本数: {len(train_ds)}")
    
    with timer("加载验证数据集"):
        val_ds = AlpamayoAVDataset(cfg.data.val)
        print(f"  验证样本数: {len(val_ds)}")

    # Create AVWorker (rollout worker)
    with timer("创建Rollout Worker"):
        rollout_placement_strategy = component_placement.get_strategy("rollout")
        rollout_group = AVWorker.create_group(cfg, component_placement).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=rollout_placement_strategy,
        )

    # Create Actor worker
    with timer("创建Actor Worker"):
        actor_worker_cls = get_actor_worker(cfg)
        actor_placement_strategy = component_placement.get_strategy("actor")
        actor_group = actor_worker_cls.create_group(cfg, component_placement).launch(
            cluster,
            name=cfg.actor.group_name,
            placement_strategy=actor_placement_strategy,
        )

    # Create runner
    with timer("创建Runner"):
        runner = AVRunner(
            cfg=cfg,
            placement=component_placement,
            train_dataset=train_ds,
            val_dataset=val_ds,
            rollout=rollout_group,
            inference=None,  # No separate inference worker for GRPO
            actor=actor_group,
            reward=None,  # AVWorker computes rewards internally
            scheduler=None,
        )

    # Initialize workers (包含模型加载)
    with timer("初始化Workers(加载模型)"):
        runner.init_workers()

    # Run training
    total_startup = time.time() - startup_time
    print(f"\n{'='*60}")
    print(f"🚀 启动完成! 总耗时: {total_startup:.1f}秒 ({total_startup/60:.1f}分钟)")
    print(f"{'='*60}\n")
    runner.run()

    print("Training completed!")


if __name__ == "__main__":
    main()
