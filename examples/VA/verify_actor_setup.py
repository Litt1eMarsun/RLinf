#!/usr/bin/env python3
"""
验证 AVFSDPActor 配置是否正确。
这个脚本会检查配置并验证正确的 Actor 类是否被选择。
"""

import hydra
import sys
from omegaconf import DictConfig, OmegaConf
from rlinf.workers.actor import get_actor_worker


@hydra.main(version_base="1.1", config_path="config", config_name="alpamayo_r1_grpo")
def verify_setup(cfg: DictConfig):
    """验证 actor worker 配置"""
    
    print("=" * 80)
    print("验证 AVFSDPActor 配置")
    print("=" * 80)
    
    # 打印关键配置
    print("\n关键配置:")
    print(f"  runner.task_type: {cfg.runner.task_type}")
    print(f"  rollout.generation_backend: {cfg.rollout.generation_backend}")
    print(f"  rollout.pipeline_stage_num: {cfg.rollout.get('pipeline_stage_num', 'NOT SET')}")
    print(f"  actor.model.model_type: {cfg.actor.model.model_type}")
    print(f"  actor.model.model_path: {cfg.actor.model.model_path}")
    print(f"  actor.tokenizer.tokenizer_model: {cfg.actor.tokenizer.tokenizer_model}")
    print(f"  actor.training_backend: {cfg.actor.training_backend}")
    
    # 获取 actor worker 类
    print("\n" + "=" * 80)
    print("获取 Actor Worker 类...")
    actor_worker_cls = get_actor_worker(cfg)
    print(f"✓ 选择的 Actor 类: {actor_worker_cls.__name__}")
    
    # 验证是否为正确的类
    expected_class = "AVFSDPActor"
    if actor_worker_cls.__name__ == expected_class:
        print(f"✓ 成功！使用的是 {expected_class}")
    else:
        print(f"✗ 错误！预期使用 {expected_class}，但实际使用的是 {actor_worker_cls.__name__}")
        return 1
    
    # 检查配置完整性
    print("\n" + "=" * 80)
    print("检查配置完整性...")
    
    issues = []
    
    # 检查必要配置
    if cfg.runner.task_type != "av":
        issues.append(f"runner.task_type 应该是 'av'，当前是 '{cfg.runner.task_type}'")
    
    if cfg.rollout.generation_backend != "av":
        issues.append(f"rollout.generation_backend 应该是 'av'，当前是 '{cfg.rollout.generation_backend}'")
    
    if not cfg.rollout.get("pipeline_stage_num"):
        issues.append("rollout.pipeline_stage_num 未设置")
    
    if cfg.actor.tokenizer.tokenizer_model != "Qwen/Qwen3-VL-2B-Instruct":
        issues.append(f"actor.tokenizer.tokenizer_model 应该是 'Qwen/Qwen3-VL-2B-Instruct'")
    
    # 检查是否有 env 配置（不应该有）
    if "env" in cfg and "group_name" in cfg.env:
        issues.append("配置中不应该包含 env.group_name（AVFSDPActor 不需要）")
    
    if issues:
        print("✗ 发现以下问题:")
        for issue in issues:
            print(f"  - {issue}")
        return 1
    else:
        print("✓ 所有配置检查通过!")
    
    # 验证 AVFSDPActor 类的特性
    print("\n" + "=" * 80)
    print("验证 AVFSDPActor 类特性...")
    
    # 检查类文档
    if actor_worker_cls.__doc__:
        print(f"类文档:")
        for line in actor_worker_cls.__doc__.strip().split('\n'):
            print(f"  {line}")
    
    # 检查 __init__ 签名
    import inspect
    init_sig = inspect.signature(actor_worker_cls.__init__)
    print(f"\n__init__ 参数: {list(init_sig.parameters.keys())}")
    
    # 检查关键方法
    key_methods = [
        'model_provider_func',
        'sync_model_to_rollout',
        'recv_rollout_trajectories',
        'run_training',
        'compute_advantages_and_returns'
    ]
    
    print("\n关键方法检查:")
    for method in key_methods:
        if hasattr(actor_worker_cls, method):
            print(f"  ✓ {method}")
        else:
            print(f"  ✗ {method} (缺失)")
    
    print("\n" + "=" * 80)
    print("✓ 验证完成！AVFSDPActor 配置正确！")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(verify_setup())

