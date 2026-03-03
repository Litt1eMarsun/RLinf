#!/usr/bin/env python3
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
可视化脚本：加载8个数据样本进行 autoregressive rollout 推理，并可视化预测轨迹与 GT 轨迹的对比。

使用 predict_action_batch（AR token 解码路径），已修复两个关键 bug：
  Bug 1：token 偏移未减去 traj_token_start_idx → 解码结果放大 ~150 倍 → 1014m ADE
  Bug 2：无条件 LogitsProcessor → VLM 在 traj_future_start 之后仍生成文字 token

用法:
    cd /home/sunyujie/workspace/RLinf
    HF_HOME=/home/sunyujie/workspace/RLinf/data/hub \\
    HUGGING_FACE_HUB_TOKEN=<your_token> \\
    python examples/visualize_rollout.py --output rollout_vis.png
"""

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader
from omegaconf import OmegaConf

from rlinf.data.datasets.alpamayo_av import AlpamayoAVDataset
from rlinf.models import get_model

_DEFAULT_CHECKPOINT_DIR = (
    "/data/workspace/results/alpamayo_r1_grpo_av_overfitting"
    "/checkpoints/global_step_40/actor"
)


def load_rl_checkpoint(model: torch.nn.Module, checkpoint_dir: str) -> None:
    """
    将 FSDP DCP 格式 checkpoint 加载到单 GPU 推理模型中。

    训练时通过 8 个 GPU 以 FSDP2 分布式格式保存，每个 .distcp 文件是一个分片。
    加载时不需要初始化 torch.distributed：dcp.load 在无进程组时会自动把所有分片
    合并成完整张量，加载到当前进程。

    checkpoint_dir 结构示例：
      global_step_40/actor/
        └── dcp_checkpoint/
              ├── __0_0.distcp
              ├── ...
              └── __7_0.distcp

    Args:
        model: 已用 get_model 加载好的 SFT 基础模型（未 FSDP 包装）。
        checkpoint_dir: 包含 dcp_checkpoint/ 子目录的 actor 目录。
    """
    dcp_dir = os.path.join(checkpoint_dir, "dcp_checkpoint")
    if not os.path.isdir(dcp_dir):
        raise FileNotFoundError(
            f"找不到 dcp_checkpoint 目录：{dcp_dir}\n"
            f"请确认 --checkpoint_dir 指向 actor/ 目录（如 global_step_40/actor）"
        )

    print(f"  读取 DCP metadata：{dcp_dir}")
    reader = FileSystemReader(dcp_dir)
    metadata = reader.read_metadata()

    # 只加载模型参数，跳过 optimizer / lr_scheduler / rng 等
    prefix = "fsdp_checkpoint.model."
    model_keys = {
        k: v for k, v in metadata.state_dict_metadata.items()
        if k.startswith(prefix)
    }
    print(f"  发现模型参数 key 数：{len(model_keys)}  "
          f"（总 checkpoint key 数：{len(metadata.state_dict_metadata)}）")

    # 预分配目标张量（每个 key 对应完整 tensor，而非分片）
    sd: dict = {}
    for k, v in model_keys.items():
        sd[k] = torch.empty(v.size, dtype=v.properties.dtype)

    # 无分布式环境下 dcp.load 自动把 8 个分片合并为完整张量
    print("  合并 8 个分片并加载参数（约 20 GB，请稍候）...")
    dcp.load(sd, checkpoint_id=dcp_dir)

    # 去掉前缀，载入模型
    model_state_dict = {k[len(prefix):]: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(model_state_dict, strict=True)
    if missing:
        print(f"  ⚠️  缺失 key（{len(missing)} 个）：{missing[:3]}...")
    if unexpected:
        print(f"  ⚠️  多余 key（{len(unexpected)} 个）：{unexpected[:3]}...")
    print(f"  ✓ RL checkpoint 加载完成（{len(model_state_dict)} 个参数）")

    del sd  # 释放临时内存


def rotate_90cc(xy):
    """将 (x, y) 逆时针旋转90度 → (-y, x)"""
    return np.stack([-xy[1], xy[0]], axis=0)


def collate_batch(batch_data_list):
    """与 av_worker._collate_batch 逻辑相同：stack 张量，list 保持原样。"""
    batch = {}
    for key in batch_data_list[0].keys():
        if isinstance(batch_data_list[0][key], torch.Tensor):
            batch[key] = torch.stack([item[key] for item in batch_data_list], dim=0)
        else:
            batch[key] = [item[key] for item in batch_data_list]
    return batch


def get_keypoint_indices(pred_length):
    """与 av_worker._compute_rewards 中的 selected_idx 完全一致。"""
    selected_gap = pred_length // 4
    return [(i + 1) * selected_gap - 1 for i in range(4)]


def visualize(pred_xyz, gt_xyz, clip_ids, output_path, weight_label="SFT weights"):
    """
    2行×4列子图：蓝=rollout，红=GT，圆点=关键点 K1~K4。
    坐标系：XY 平面，逆时针旋转90°（与 inference.ipynb 一致）。
    """
    B = pred_xyz.shape[0]
    T_fut = pred_xyz.shape[1]
    selected_idx = get_keypoint_indices(T_fut)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for b in range(B):
        ax = axes[b]

        p = pred_xyz[b].cpu().numpy()     # [T_fut, 3]
        g = gt_xyz[b, 0].cpu().numpy()    # [T_fut, 3]

        p_rot = rotate_90cc(p[:, :2].T)  # [2, T_fut]
        g_rot = rotate_90cc(g[:, :2].T)

        pk = rotate_90cc(p[selected_idx, :2].T)  # [2, 4]
        gk = rotate_90cc(g[selected_idx, :2].T)

        ax.plot(p_rot[0], p_rot[1], "b-", lw=2, label="Rollout (AR)", alpha=0.8)
        ax.plot(g_rot[0], g_rot[1], "r-", lw=2, label="GT", alpha=0.8)
        ax.scatter(pk[0], pk[1], c="blue", s=120, edgecolors="darkblue", lw=2, zorder=5)
        ax.scatter(gk[0], gk[1], c="red",  s=120, edgecolors="darkred",  lw=2, zorder=5)

        for k in range(4):
            ax.annotate(f"K{k+1}",
                        xy=(pk[0, k], pk[1, k]), xytext=(5, 5),
                        textcoords="offset points", fontsize=9, color="blue", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="blue"))
            ax.annotate(f"K{k+1}",
                        xy=(gk[0, k], gk[1, k]), xytext=(5, -15),
                        textcoords="offset points", fontsize=9, color="red", weight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="red"))

        # Keypoint ADE（与 av_worker._compute_rewards 一致）
        kp_ade = torch.norm(
            pred_xyz[b, selected_idx] - gt_xyz[b, 0, selected_idx], dim=-1
        ).mean().item()

        clip_short = clip_ids[b][:8] + "…" if len(clip_ids[b]) > 8 else clip_ids[b]
        ax.set_title(f"Clip: {clip_short}\nKeypoint ADE: {kp_ade:.3f} m", fontsize=10, pad=8)
        ax.set_xlabel("x (m)", fontsize=9)
        ax.set_ylabel("y (m)", fontsize=9)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    for b in range(B, len(axes)):
        axes[b].set_visible(False)

    plt.suptitle(f"Rollout (AR token decode) vs GT  [{weight_label}]", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"可视化结果已保存到: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="可视化 AR rollout 轨迹与 GT 轨迹对比")
    parser.add_argument("--model_path", type=str, default="nvidia/Alpamayo-R1-10B")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=_DEFAULT_CHECKPOINT_DIR,
        help="RL 训练保存的 actor checkpoint 目录（含 dcp_checkpoint/ 子目录）。"
             "设为空字符串 '' 则使用原始 SFT 权重（不加载 RL checkpoint）。"
             f"默认：{_DEFAULT_CHECKPOINT_DIR}",
    )
    parser.add_argument("--data_path", type=str,
                        default="/home/sunyujie/workspace/RLinf/data/clip_index.parquet")
    parser.add_argument("--cache_dir", type=str,
                        default="/home/sunyujie/workspace/RLinf/data")
    parser.add_argument("--output", type=str, default="rollout_visualization.png")
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--num_history_steps", type=int, default=16)
    parser.add_argument("--num_future_steps", type=int, default=64)
    parser.add_argument("--time_step", type=float, default=0.1)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--initial_timestamp_us", type=int, default=5100000)
    parser.add_argument("--precision", type=str, default="bf16")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.98)
    parser.add_argument("--max_new_tokens", type=int, default=0,
                        help="0 表示使用默认值（256 + 1 + tokens_per_traj + 1）")
    args = parser.parse_args()

    print("=" * 80)
    print("Rollout 轨迹可视化脚本  [AR token 解码路径，已修复 token 偏移 bug]")
    print("=" * 80)

    # ── 1. 加载数据 ────────────────────────────────────────────────────────────
    print("\n[1/4] 加载数据集...")
    dataset_config = OmegaConf.create({
        "data_path": args.data_path,
        "cache_dir": args.cache_dir,
        "num_history_steps": args.num_history_steps,
        "num_future_steps": args.num_future_steps,
        "time_step": args.time_step,
        "num_frames": args.num_frames,
        "maybe_stream": False,
        "initial_timestamp_us": args.initial_timestamp_us,
        "max_samples": args.num_samples,
    })

    dataset = AlpamayoAVDataset(dataset_config)
    print(f"  ✓ 数据集加载完成，共 {len(dataset)} 个可用样本")

    batch_data_list = []
    clip_ids = []
    for i in range(min(args.num_samples, len(dataset))):
        print(f"  加载样本 {i+1}/{args.num_samples}...")
        sample = dataset[i]
        batch_data_list.append(sample)
        clip_ids.append(sample["clip_id"])
    print(f"  ✓ 共加载 {len(batch_data_list)} 个样本")

    batch = collate_batch(batch_data_list)

    # ── 2. 加载模型 ────────────────────────────────────────────────────────────
    use_rl_ckpt = bool(args.checkpoint_dir)
    weight_tag = f"RL  [{args.checkpoint_dir}]" if use_rl_ckpt else "SFT（原始权重）"
    print(f"\n[2/4] 加载模型: {args.model_path}")
    print(f"  权重来源: {weight_tag}")

    model_config = OmegaConf.create({
        "model_type": "alpamayo",
        "model_path": args.model_path,
        "precision": args.precision,
        "is_lora": False,
    })
    model = get_model(model_config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if use_rl_ckpt:
        print(f"\n  [2b] 加载 RL checkpoint: {args.checkpoint_dir}")
        load_rl_checkpoint(model, args.checkpoint_dir)

    model.eval()
    print(f"  ✓ 模型加载完成，设备: {device}")

    # 打印轨迹 token 诊断信息
    traj_start_id = model.traj_future_start_id
    traj_end_id   = model.traj_future_end_id
    traj_offset   = model.future_token_start_idx
    traj_vsize    = model.traj_tokenizer.vocab_size
    action_dims   = model.traj_tokenizer.action_space.get_action_space_dims()
    tokens_per_traj = 1
    for d in action_dims:
        tokens_per_traj *= d
    print(f"  traj_future_start_id : {traj_start_id}")
    print(f"  traj_future_end_id   : {traj_end_id}")
    print(f"  traj_token_offset    : {traj_offset}  (<i0> 的绝对词表 ID)")
    print(f"  traj_vocab_size      : {traj_vsize}   (DiscreteTrajectoryTokenizer.num_bins)")
    print(f"  action_space_dims    : {action_dims}")
    print(f"  tokens_per_traj      : {tokens_per_traj}")

    # ── 3. 移动数据到设备 ──────────────────────────────────────────────────────
    print(f"\n[3/4] 运行 AR rollout 推理...")
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    kwargs = {
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if args.max_new_tokens > 0:
        kwargs["max_new_tokens"] = args.max_new_tokens

    with torch.no_grad(), torch.autocast(device.type, dtype=torch.bfloat16):
        _, result = model.predict_action_batch(batch, mode="train", **kwargs)

    pred_xyz = result["pred_xyz"]   # [B, T_fut, 3]
    gt_xyz   = result["gt_xyz"]     # [B, 1, T_fut, 3]

    print(f"  ✓ 推理完成")
    print(f"    pred_xyz shape : {pred_xyz.shape}")
    print(f"    gt_xyz   shape : {gt_xyz.shape}")

    # 计算奖励（与 av_worker._compute_rewards 一致）
    selected_idx = get_keypoint_indices(pred_xyz.shape[1])
    pred_kp = pred_xyz[:, selected_idx, :]           # [B, 4, 3]
    gt_kp   = gt_xyz[:, 0, selected_idx, :]          # [B, 4, 3]
    ade = torch.norm(pred_kp - gt_kp, dim=-1).mean(dim=-1)  # [B]
    rewards = -ade
    print(f"  关键点索引    : {selected_idx}")
    print(f"  平均 ADE      : {ade.mean().item():.3f} m")
    print(f"  奖励均值/范围 : {rewards.mean().item():.3f}  "
          f"[{rewards.min().item():.3f}, {rewards.max().item():.3f}]")

    # ── 4. 可视化 ──────────────────────────────────────────────────────────────
    print(f"\n[4/4] 生成可视化...")
    weight_label = (
        f"RL  step={os.path.basename(os.path.dirname(args.checkpoint_dir))}"
        if use_rl_ckpt else "SFT weights"
    )
    visualize(
        pred_xyz=pred_xyz,
        gt_xyz=gt_xyz,
        clip_ids=clip_ids,
        output_path=args.output,
        weight_label=weight_label,
    )

    print("\n" + "=" * 80)
    print("完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
