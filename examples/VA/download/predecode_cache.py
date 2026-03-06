#!/usr/bin/env python3
"""
预解码缓存脚本：从 HuggingFace Hub 流式(或本地)读取 PhysicalAI-AV 数据，
按 chunk 批量解码视频帧为 JPEG、egomotion 存为 npz。

核心优化：按 chunk 批量处理
  - 同一 chunk 的 zip (~2GB/camera) 只下载/打开一次
  - 一次性解码该 chunk 内所有 clip → 避免重复下载
  - 流式模式下 zip 数据只在内存中，不保留到本地磁盘

数据流:
  HF Hub (stream) / 本地 zip → 打开 chunk zip → 遍历 chunk 内所有 clip
  → 解码帧 → 存 JPEG + npz → 关闭 zip → 释放内存 → 下一个 chunk

缓存目录结构：
  {output_dir}/{clip_id}/
    egomotion.npz
    video_meta.npz
    camera_cross_left_120fov/frame_{0..3}.jpg
    camera_front_wide_120fov/frame_{0..3}.jpg
    camera_cross_right_120fov/frame_{0..3}.jpg
    camera_front_tele_30fov/frame_{0..3}.jpg

用法:
    cd /home/sunyujie/workspace/RLinf
    source .venv/bin/activate

    # 流式模式（不保留 zip，推荐磁盘空间不足时）
    python examples/VA/predecode_cache.py \\
        --output-dir /data/workspace/RLinf/decoded_cache \\
        --max-clips 6400 --stream --resume

    # 本地模式（需已下载 zip 到 cache-dir）
    python examples/VA/predecode_cache.py \\
        --output-dir /data/workspace/RLinf/decoded_cache \\
        --max-clips 6400 --no-stream --resume

    # 只处理前 100 个 clip 测试
    python examples/VA/predecode_cache.py \\
        --max-clips 100 --stream --resume
"""

import argparse
import gc
import io
import os
import sys
import time
import traceback
import zipfile
from collections import defaultdict

import numpy as np

# ── 常量 ────────────────────────────────────────────────────────────────────
DEFAULT_CAMERAS = [
    "camera_cross_left_120fov",
    "camera_front_wide_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
]

CAMERA_NAME_TO_INDEX = {
    "camera_cross_left_120fov": 0,
    "camera_front_wide_120fov": 1,
    "camera_cross_right_120fov": 2,
    "camera_rear_left_70fov": 3,
    "camera_rear_tele_30fov": 4,
    "camera_rear_right_70fov": 5,
    "camera_front_tele_30fov": 6,
}


def _compute_egomotion_for_clip(
    egomotion_interpolator,
    num_history_steps: int,
    num_future_steps: int,
    time_step: float,
    initial_timestamp_us: int,
) -> dict[str, np.ndarray]:
    """从 egomotion interpolator 计算局部坐标系下的轨迹。"""
    import scipy.spatial.transform as spt

    t0_us = initial_timestamp_us

    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    ego_history = egomotion_interpolator(history_timestamps)
    ego_history_xyz = ego_history.pose.translation
    ego_history_quat = ego_history.pose.rotation.as_quat()
    ego_future = egomotion_interpolator(future_timestamps)
    ego_future_xyz = ego_future.pose.translation
    ego_future_quat = ego_future.pose.rotation.as_quat()

    t0_xyz = ego_history_xyz[-1].copy()
    t0_quat = ego_history_quat[-1].copy()
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    return {
        "history_xyz": t0_rot_inv.apply(ego_history_xyz - t0_xyz).astype(np.float32),
        "history_rot": (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat))
        .as_matrix()
        .astype(np.float32),
        "future_xyz": t0_rot_inv.apply(ego_future_xyz - t0_xyz).astype(np.float32),
        "future_rot": (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat))
        .as_matrix()
        .astype(np.float32),
    }


def decode_one_chunk(
    chunk_id: int,
    clip_ids: list[str],
    output_dir: str,
    cache_dir: str,
    num_history_steps: int,
    num_future_steps: int,
    time_step: float,
    num_frames: int,
    initial_timestamp_us: int,
    jpeg_quality: int,
    cameras: list[str],
    use_stream: bool,
) -> tuple[int, int, int, list[str]]:
    """
    解码一整个 chunk 的所有 clip。

    关键优化：每个 chunk zip 只打开/下载一次，遍历其中所有 clip。
    流式模式下 zip 内容只存在于内存，处理完释放。

    Returns:
        (chunk_id, success_count, fail_count, failed_clip_ids)
    """
    import pandas as pd
    import physical_ai_av
    from physical_ai_av import egomotion as ego_module
    from physical_ai_av import video as video_module
    from PIL import Image

    avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=cache_dir)
    t0_us = initial_timestamp_us

    image_timestamps = np.array(
        [
            t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000)
            for i in range(num_frames)
        ],
        dtype=np.int64,
    )

    # 过滤已完成的 clip
    pending_clips = []
    for cid in clip_ids:
        done_marker = os.path.join(output_dir, cid, ".done")
        if not os.path.exists(done_marker):
            pending_clips.append(cid)

    if not pending_clips:
        return (chunk_id, len(clip_ids), 0, [])

    success_count = 0
    fail_count = 0
    failed_clips = []

    # 为所有 pending clip 创建目录
    for cid in pending_clips:
        os.makedirs(os.path.join(output_dir, cid), exist_ok=True)

    # ── 1. Egomotion: 打开 chunk zip 一次，提取所有 clip ────────────
    ego_feature = avdi.features.LABELS.EGOMOTION
    ego_chunk_file = avdi.features.get_chunk_feature_filename(chunk_id, ego_feature)

    ego_data_map = {}  # clip_id → egomotion dict
    try:
        with avdi.open_file(ego_chunk_file, maybe_stream=use_stream) as f:
            with zipfile.ZipFile(f, "r") as zf:
                for cid in pending_clips:
                    try:
                        clip_files = avdi.features.get_clip_files_in_zip(cid, ego_feature)
                        ego_df = pd.read_parquet(
                            io.BytesIO(zf.read(clip_files["egomotion"]))
                        )
                        interpolator = ego_module.EgomotionState.from_egomotion_df(
                            ego_df
                        ).create_interpolator(ego_df["timestamp"].to_numpy())

                        ego_result = _compute_egomotion_for_clip(
                            interpolator,
                            num_history_steps,
                            num_future_steps,
                            time_step,
                            initial_timestamp_us,
                        )
                        ego_data_map[cid] = ego_result

                        # 保存 egomotion
                        np.savez_compressed(
                            os.path.join(output_dir, cid, "egomotion.npz"),
                            **ego_result,
                        )
                    except Exception as e:
                        failed_clips.append(cid)
                        fail_count += 1
                        print(f"    ✗ ego {cid[:12]}...: {e}")
    except Exception as e:
        print(f"    ✗ 无法打开 egomotion chunk {chunk_id}: {e}")
        return (chunk_id, 0, len(pending_clips), [c for c in pending_clips])

    del ego_data_map
    gc.collect()

    # 过滤掉 egomotion 失败的 clip
    active_clips = [c for c in pending_clips if c not in failed_clips]

    # ── 2. Camera frames: 每个 camera 打开 chunk zip 一次 ──────────
    # 为每个 clip 初始化时间戳收集器
    clip_cam_meta = {cid: {"indices": [], "abs_ts": []} for cid in active_clips}

    for cam_name in cameras:
        cam_feature = getattr(avdi.features.CAMERA, cam_name.upper())
        cam_chunk_file = avdi.features.get_chunk_feature_filename(chunk_id, cam_feature)
        cam_idx = CAMERA_NAME_TO_INDEX.get(cam_name, 0)

        try:
            with avdi.open_file(cam_chunk_file, maybe_stream=use_stream) as f:
                with zipfile.ZipFile(f, "r") as zf:
                    for cid in active_clips:
                        try:
                            clip_files = avdi.features.get_clip_files_in_zip(
                                cid, cam_feature
                            )
                            video_data = io.BytesIO(zf.read(clip_files["video"]))
                            ts_data = pd.read_parquet(
                                io.BytesIO(zf.read(clip_files["frame_timestamps"]))
                            )["timestamp"].to_numpy()

                            reader = video_module.SeekVideoReader(
                                video_data=video_data, timestamps=ts_data
                            )
                            frames, frame_ts = reader.decode_images_from_timestamps(
                                image_timestamps
                            )
                            reader.close()
                            del reader, video_data

                            # 保存 JPEG
                            cam_dir = os.path.join(output_dir, cid, cam_name)
                            os.makedirs(cam_dir, exist_ok=True)
                            for fi in range(frames.shape[0]):
                                img = Image.fromarray(frames[fi])
                                img.save(
                                    os.path.join(cam_dir, f"frame_{fi}.jpg"),
                                    "JPEG",
                                    quality=jpeg_quality,
                                )
                            del frames

                            clip_cam_meta[cid]["indices"].append(cam_idx)
                            clip_cam_meta[cid]["abs_ts"].append(
                                frame_ts.astype(np.int64)
                            )

                        except Exception as e:
                            if cid not in failed_clips:
                                failed_clips.append(cid)
                                fail_count += 1
                            print(f"    ✗ {cam_name} {cid[:12]}...: {e}")

        except Exception as e:
            print(f"    ✗ 无法打开 {cam_name} chunk {chunk_id}: {e}")
            # 标记所有未处理的 clip 为失败
            for cid in active_clips:
                if cid not in failed_clips:
                    failed_clips.append(cid)
                    fail_count += 1
            break

        gc.collect()

    # ── 3. 保存 video metadata + 完成标记 ─────────────────────────
    for cid in active_clips:
        if cid in failed_clips:
            continue

        meta = clip_cam_meta[cid]
        if len(meta["indices"]) != len(cameras):
            # 某个 camera 失败了，跳过
            if cid not in failed_clips:
                failed_clips.append(cid)
                fail_count += 1
            continue

        all_abs_ts = np.stack(meta["abs_ts"], axis=0)
        all_cam_idx = np.array(meta["indices"], dtype=np.int64)
        camera_tmin = all_abs_ts.min()
        all_rel_ts = (all_abs_ts - camera_tmin).astype(np.float32) * 1e-6

        np.savez_compressed(
            os.path.join(output_dir, cid, "video_meta.npz"),
            camera_indices=all_cam_idx,
            absolute_timestamps=all_abs_ts,
            relative_timestamps=all_rel_ts,
            camera_names=np.array(cameras),
        )

        with open(os.path.join(output_dir, cid, ".done"), "w") as f:
            f.write("ok\n")
        success_count += 1

    return (chunk_id, success_count, fail_count, failed_clips)


def main():
    parser = argparse.ArgumentParser(
        description="预解码 PhysicalAI-AV clip 到 JPEG + npz 缓存（按 chunk 批量处理）"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data/workspace/RLinf/data/clip_index.parquet",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/data/workspace/RLinf/data",
        help="HuggingFace Hub 缓存目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/workspace/RLinf/decoded_cache",
        help="解码缓存输出目录",
    )
    parser.add_argument(
        "--max-clips", type=int, default=6400, help="最多处理多少个 clip"
    )
    parser.add_argument(
        "--start-clip",
        type=int,
        default=0,
        help="从第几个 clip 开始（与 --max-clips 配合实现分段处理）",
    )
    parser.add_argument(
        "--jpeg-quality", type=int, default=85, help="JPEG 压缩质量 (1-100)"
    )
    parser.add_argument("--num-history-steps", type=int, default=16)
    parser.add_argument("--num-future-steps", type=int, default=64)
    parser.add_argument("--time-step", type=float, default=0.1)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--initial-timestamp-us", type=int, default=5_100_000)

    stream_group = parser.add_mutually_exclusive_group()
    stream_group.add_argument(
        "--stream",
        action="store_true",
        default=True,
        help="从 HF Hub 流式读取 zip（不保留本地 zip，默认）",
    )
    stream_group.add_argument(
        "--no-stream",
        action="store_true",
        help="从本地已下载的 zip 缓存读取（更快，需已有本地 zip）",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="跳过已存在 .done 标记的 clip（推荐始终开启）",
    )
    args = parser.parse_args()

    use_stream = not args.no_stream

    os.environ["HF_HOME"] = args.cache_dir
    os.environ["HF_HUB_CACHE"] = os.path.join(args.cache_dir, "hub")

    mode_label = "流式 stream（不保留 zip）" if use_stream else "本地 local zip"

    print(f"\n{'='*80}")
    print(f"PhysicalAI-AV 预解码缓存工具（chunk 批量模式）")
    print(f"{'='*80}")
    print(f"  数据源模式  : {mode_label}")
    print(f"  数据索引    : {args.data_path}")
    print(f"  缓存目录    : {args.cache_dir}")
    print(f"  输出目录    : {args.output_dir}")
    print(f"  clip 范围   : [{args.start_clip}, {args.start_clip + args.max_clips})")
    print(f"  JPEG 质量   : {args.jpeg_quality}")
    print(f"  Resume      : {args.resume}")
    if use_stream:
        print(f"  ⚡ 流式模式: 每个 chunk zip 只流式下载一次，处理完即释放内存")
        print(f"     同一 chunk 的 ~100 clip 共享一次下载，避免重复传输")
    print(f"{'='*80}\n")

    import pandas as pd

    # 加载 clip IDs
    df = pd.read_parquet(args.data_path)
    df_r = df.reset_index()
    train = df_r[df_r["split"] == "train"]
    clip_ids = train["clip_id"].tolist()[
        args.start_clip : args.start_clip + args.max_clips
    ]
    print(f"✓ 选取了 {len(clip_ids)} 个 train clip")

    # 按 chunk 分组
    clip_to_chunk = dict(zip(train["clip_id"], train["chunk"]))
    chunks = defaultdict(list)
    for cid in clip_ids:
        chunks[clip_to_chunk[cid]].append(cid)

    chunk_ids_sorted = sorted(chunks.keys())
    print(f"  涉及 {len(chunk_ids_sorted)} 个 chunk "
          f"(range [{chunk_ids_sorted[0]}, {chunk_ids_sorted[-1]}])")

    # 如果 resume，统计已完成的
    if args.resume:
        already_done = sum(
            1
            for cid in clip_ids
            if os.path.exists(os.path.join(args.output_dir, cid, ".done"))
        )
        remaining = len(clip_ids) - already_done
        print(f"  已完成: {already_done}，剩余: {remaining}")
        if remaining == 0:
            print("所有 clip 已解码完成！")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    # ── 按 chunk 顺序处理 ───────────────────────────────────────────
    total_success = 0
    total_fail = 0
    start_time = time.time()

    for ci, chunk_id in enumerate(chunk_ids_sorted, 1):
        chunk_clips = chunks[chunk_id]

        # 快速检查是否全部已完成
        pending_count = sum(
            1
            for cid in chunk_clips
            if not os.path.exists(os.path.join(args.output_dir, cid, ".done"))
        )
        if pending_count == 0:
            total_success += len(chunk_clips)
            continue

        elapsed = time.time() - start_time
        speed = (total_success + total_fail) / elapsed if elapsed > 0 else 0
        remaining_clips = len(clip_ids) - total_success - total_fail
        eta = remaining_clips / speed / 60 if speed > 0 else 0

        print(
            f"\n[chunk {ci}/{len(chunk_ids_sorted)}] "
            f"chunk_id={chunk_id}, "
            f"{len(chunk_clips)} clips ({pending_count} pending) "
            f"| 进度: {total_success}/{len(clip_ids)} "
            f"| ETA: {eta:.1f}min"
        )

        try:
            chunk_id_result, s, f, failed = decode_one_chunk(
                chunk_id=chunk_id,
                clip_ids=chunk_clips,
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                num_history_steps=args.num_history_steps,
                num_future_steps=args.num_future_steps,
                time_step=args.time_step,
                num_frames=args.num_frames,
                initial_timestamp_us=args.initial_timestamp_us,
                jpeg_quality=args.jpeg_quality,
                cameras=DEFAULT_CAMERAS,
                use_stream=use_stream,
            )
            total_success += s
            total_fail += f
            # 跳过的（已完成）也算
            skipped = len(chunk_clips) - pending_count
            total_success += skipped

            if f > 0:
                print(f"  chunk {chunk_id}: ✓{s} ✗{f}")
            else:
                print(f"  chunk {chunk_id}: ✓{s + skipped} clips done")

        except KeyboardInterrupt:
            print("\n\n中断！已完成的 clip 缓存会保留，下次用 --resume 继续。")
            break
        except Exception as e:
            print(f"  chunk {chunk_id}: 异常: {e}")
            total_fail += len(chunk_clips)
            traceback.print_exc()

        gc.collect()

    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"预解码完成")
    print(f"{'='*80}")
    print(f"  模式  : {mode_label}")
    print(f"  成功  : {total_success}/{len(clip_ids)}")
    print(f"  失败  : {total_fail}")
    print(f"  耗时  : {elapsed:.1f}s ({elapsed/60:.1f}min)")
    if elapsed > 0 and (total_success + total_fail) > 0:
        print(f"  速度  : {(total_success + total_fail)/elapsed:.1f} clips/s")
    print(f"  输出  : {args.output_dir}")

    # 统计实际缓存大小
    try:
        total_size = 0
        done_count = 0
        for cid in clip_ids:
            clip_dir = os.path.join(args.output_dir, cid)
            if os.path.exists(os.path.join(clip_dir, ".done")):
                done_count += 1
                for root, _, files in os.walk(clip_dir):
                    for fname in files:
                        total_size += os.path.getsize(os.path.join(root, fname))
        print(f"  缓存  : {done_count} clips, {total_size/1024/1024/1024:.2f} GB")
    except Exception:
        pass

    print(f"{'='*80}")

    if total_fail > 0:
        print(f"\n⚠️  有 {total_fail} 个 clip 失败，可以用 --resume 重试")
        sys.exit(1)


if __name__ == "__main__":
    main()
