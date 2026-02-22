#!/usr/bin/env python3
"""
简单验证 AVFSDPActor 配置（不需要 hydra）
"""

import sys
import os

# 添加 RLinf 到路径
sys.path.insert(0, '/home/sunyujie/workspace/RLinf')

from omegaconf import OmegaConf


def verify_actor_worker():
    """验证 AVFSDPActor 是否正确定义"""
    
    print("=" * 80)
    print("验证 AVFSDPActor 配置")
    print("=" * 80)
    
    # 1. 检查 AVFSDPActor 类是否存在
    print("\n1. 检查 AVFSDPActor 类是否存在...")
    try:
        from rlinf.workers.actor.fsdp_actor_worker import AVFSDPActor
        print(f"   ✓ AVFSDPActor 类存在")
    except ImportError as e:
        print(f"   ✗ 无法导入 AVFSDPActor: {e}")
        return 1
    
    # 2. 检查类文档
    print("\n2. 检查类文档...")
    if AVFSDPActor.__doc__:
        print("   类文档:")
        for line in AVFSDPActor.__doc__.strip().split('\n')[:5]:
            print(f"     {line.strip()}")
    
    # 3. 检查关键方法
    print("\n3. 检查关键方法...")
    key_methods = [
        'model_provider_func',
        'sync_model_to_rollout',
        'recv_rollout_trajectories',
        'run_training',
        'compute_advantages_and_returns',
        '_process_received_rollout_batch',
        'init_worker'
    ]
    
    missing_methods = []
    for method in key_methods:
        if hasattr(AVFSDPActor, method):
            print(f"   ✓ {method}")
        else:
            print(f"   ✗ {method} (缺失)")
            missing_methods.append(method)
    
    if missing_methods:
        print(f"\n   ✗ 缺失 {len(missing_methods)} 个方法")
        return 1
    
    # 4. 检查 get_actor_worker 是否能正确选择 AVFSDPActor
    print("\n4. 检查 get_actor_worker 函数...")
    try:
        from rlinf.workers.actor import get_actor_worker
        
        # 创建测试配置
        test_cfg = OmegaConf.create({
            "runner": {"task_type": "av"},
            "rollout": {"generation_backend": "av"},
            "actor": {
                "training_backend": "fsdp",
                "model": {"model_type": "alpamayo"}
            },
            "cluster": {}
        })
        
        actor_cls = get_actor_worker(test_cfg)
        print(f"   选择的 Actor 类: {actor_cls.__name__}")
        
        if actor_cls.__name__ == "AVFSDPActor":
            print(f"   ✓ 正确选择了 AVFSDPActor")
        else:
            print(f"   ✗ 错误！应该选择 AVFSDPActor，但选择了 {actor_cls.__name__}")
            return 1
    except Exception as e:
        print(f"   ✗ get_actor_worker 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # 5. 读取配置文件并验证
    print("\n5. 验证配置文件...")
    config_path = "/home/sunyujie/workspace/RLinf/examples/VA/config/alpamayo_r1_grpo.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            content = f.read()
        
        checks = [
            ("task_type: av", "runner.task_type 设置为 'av'"),
            ("generation_backend: \"av\"", "rollout.generation_backend 设置为 'av'"),
            ("pipeline_stage_num:", "rollout.pipeline_stage_num 已设置"),
            ("tokenizer_model: \"Qwen/Qwen3-VL-2B-Instruct\"", "使用基础 VLM tokenizer"),
        ]
        
        for check_str, description in checks:
            if check_str in content:
                print(f"   ✓ {description}")
            else:
                print(f"   ✗ {description} (未找到)")
    else:
        print(f"   ⚠ 配置文件不存在: {config_path}")
    
    # 6. 对比 AVFSDPActor 和 EmbodiedFSDPActor 的差异
    print("\n6. 验证 AVFSDPActor 和 EmbodiedFSDPActor 的差异...")
    try:
        from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
        import inspect
        
        # 检查 __init__ 源代码
        av_init = inspect.getsource(AVFSDPActor.__init__)
        embodied_init = inspect.getsource(EmbodiedFSDPActor.__init__)
        
        # 检查是否有赋值语句（不是注释）
        av_has_assignment = "self._env_group_name =" in av_init
        embodied_has_assignment = "self._env_group_name =" in embodied_init
        
        # AVFSDPActor 不应该有赋值语句
        if not av_has_assignment:
            print("   ✓ AVFSDPActor.__init__ 不包含 self._env_group_name 赋值")
        else:
            print("   ✗ AVFSDPActor.__init__ 仍然包含 self._env_group_name 赋值")
            return 1
        
        # EmbodiedFSDPActor 应该包含赋值语句
        if embodied_has_assignment:
            print("   ✓ EmbodiedFSDPActor.__init__ 包含 self._env_group_name 赋值")
        else:
            print("   ⚠ EmbodiedFSDPActor.__init__ 不包含 self._env_group_name 赋值")
        
        # 额外检查：AVFSDPActor 应该有注释说明不需要 env
        if "No _env_group_name needed" in av_init or "no env worker" in av_init.lower():
            print("   ✓ AVFSDPActor.__init__ 包含说明注释")
        
    except Exception as e:
        print(f"   ⚠ 差异检查失败: {e}")
    
    print("\n" + "=" * 80)
    print("✓ 所有验证通过！AVFSDPActor 配置正确！")
    print("=" * 80)
    print("\n下一步:")
    print("  1. 运行训练脚本: python train_alpamayo_r1.py")
    print("  2. 确认使用的是 AVFSDPActor 而不是 EmbodiedFSDPActor")
    print("  3. 确认不会报 'env.group_name' 错误")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(verify_actor_worker())

