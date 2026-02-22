# PhysicalAI-AV 数据集预下载指南

## 问题背景

`AlpamayoAVDataset` 使用懒加载设计：
- `AlpamayoAVDataset(config)` 只初始化配置，不下载数据
- `dataset[0]` 才真正从 HuggingFace 下载数据
- 首次下载非常慢：
  - EGOMOTION 数据：~5分钟/clip
  - 4个相机视频：~10-30分钟/clip

## 解决方案：预下载脚本

使用 `preload_data.py` 提前下载所需的 clips 到本地缓存。

## 使用方法

### 基本用法

```bash
cd /home/sunyujie/workspace/RLinf/examples/VA

# 下载前1280个clips（包含视频）
python preload_data.py

# 只下载 egomotion 数据（快10倍）
python preload_data.py --skip-cameras
```

### 高级选项

```bash
# 自定义参数
python preload_data.py \
    --parquet /home/sunyujie/workspace/RLinf/data/clip_index.parquet \
    --cache-dir /data/workspace/RLinf/data/ \
    --max-clips 1280 \
    --skip-cameras

# 断点续传（从第100个clip继续）
python preload_data.py --start-idx 100 --skip-cameras

# 智能跳过已下载的clips（推荐！）
python preload_data.py --skip-cached --skip-cameras

# 组合使用：跳过已下载 + 从特定索引继续
python preload_data.py --skip-cached --start-idx 100 --skip-cameras
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--parquet` | `/home/sunyujie/workspace/RLinf/data/clip_index.parquet` | 包含 clip IDs 的 parquet 文件路径 |
| `--cache-dir` | `/data/workspace/RLinf/data/` | 数据缓存目录（会设置为 HF_HOME） |
| `--max-clips` | `1280` | 最多下载的 clips 数量 |
| `--skip-cameras` | `False` | 只下载 egomotion 数据，跳过相机视频 |
| `--start-idx` | `0` | 从第几个 clip 开始（用于断点续传） |
| `--skip-cached` | `False` | **智能跳过已成功下载的 clips（推荐使用！）** |

## 时间估算

### 下载完整数据（egomotion + 相机视频）
- 单个 clip：15-30 分钟
- 1280 clips：**320-640 小时**（13-27 天）

### 只下载 egomotion 数据（推荐）
- 单个 clip：3-5 分钟
- 1280 clips：**64-107 小时**（2.6-4.5 天）

**建议**：先使用 `--skip-cameras` 下载 egomotion 数据进行开发和测试。

## 🎯 重要更新：数据存储位置

### ✅ 已修复：数据现在会下载到指定目录

脚本已更新，确保数据下载到你指定的 `--cache-dir` 目录，而不是默认的 `~/.cache/huggingface/`。

**实际存储位置**：
```
/data/workspace/RLinf/data/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/
```

**工作原理**：
- 在导入 `physical_ai_av` **之前**设置环境变量
- 设置 `HF_HOME`, `HF_HUB_CACHE` 等多个环境变量
- 确保 HuggingFace 使用自定义缓存路径

## 智能缓存跟踪 (NEW! 🎉)

### 工作原理

脚本会在缓存目录中维护一个 `.downloaded_clips.txt` 文件，记录所有成功下载的 clip ID。

使用 `--skip-cached` 参数时：
- ✅ 自动跳过已成功下载的 clips
- ✅ 只下载新的或之前失败的 clips
- ✅ 支持与 `--start-idx` 组合使用
- ✅ 断电/中断后可以无缝继续

### 推荐工作流程

```bash
# 第一次运行（下载前100个clips）
python preload_data.py --skip-cached --skip-cameras --max-clips 100

# 中断后继续（自动跳过已下载的）
python preload_data.py --skip-cached --skip-cameras --max-clips 100

# 扩展到1280个clips（只下载新的clips）
python preload_data.py --skip-cached --skip-cameras --max-clips 1280

# 重试失败的clips（手动从特定索引开始）
python preload_data.py --skip-cached --start-idx 500 --max-clips 600
```

### 缓存文件位置

**跟踪文件**：
```
/data/workspace/RLinf/data/.downloaded_clips.txt
```

每行一个 clip ID，例如：
```
25cd4769-5dcf-4b53-a351-bf2c5deb6124
2edf278f-d5e3-4b83-b5df-923a04335725
...
```

**实际数据存储**：
```
/data/workspace/RLinf/data/hub/
├── datasets--nvidia--PhysicalAI-Autonomous-Vehicles/
│   ├── blobs/          # 实际数据文件（哈希命名）
│   └── snapshots/      # 版本快照
└── .locks/             # 下载锁文件
```

**注意**：HuggingFace 使用内部格式存储数据，不会创建以 clip_id 命名的文件夹。

## 使用预下载的数据

下载完成后，在训练/测试脚本中设置：

```python
import os

# 设置缓存目录
os.environ['HF_HOME'] = '/data/workspace/RLinf/data/'

# 配置 dataset
config = OmegaConf.create({
    "data_path": "/home/sunyujie/workspace/RLinf/data/clip_index.parquet",
    "num_history_steps": 16,
    "num_future_steps": 64,
    "time_step": 0.1,
    "num_frames": 4,
    "maybe_stream": False,  # 使用本地缓存
    "initial_timestamp_us": 5100000,
})

dataset = AlpamayoAVDataset(config)
```

## 故障排除

### 问题1：下载中断
使用 `--start-idx` 从上次停止的位置继续：
```bash
python preload_data.py --start-idx 500 --skip-cameras
```

### 问题2：FileNotFoundError with maybe_stream=False
说明数据还没下载到本地，需要先运行 `preload_data.py`。

### 问题3：磁盘空间不足
估算空间需求：
- 只 egomotion：~50GB（1280 clips）
- 完整数据：~500GB-1TB（1280 clips，包含视频）

检查空间：
```bash
df -h /data/workspace/RLinf/data/
```

## 监控下载进度

脚本会显示：
- 总体进度条（tqdm）
- 每个 clip 的详细状态
- 成功/失败统计
- 失败的 clip 列表

示例输出：
```
Overall progress: 45%|████████          | 576/1280 [48:00:00<48:00:00, 245.45s/it]

Preloading clip: 030c760c-ae38-49aa-9ad8-f5650a545d26
  - Downloading egomotion... ✓
  - Skipping camera data (skip_cameras=True)
✓ Clip 030c760c-ae38-49aa-9ad8-f5650a545d26 cached successfully
```

## 最佳实践

1. **先测试单个 clip**：
   ```bash
   python preload_data.py --max-clips 1 --skip-cameras
   ```

2. **小批量下载**：
   ```bash
   python preload_data.py --max-clips 10 --skip-cameras
   ```

3. **后台运行大批量任务**：
   ```bash
   nohup python preload_data.py --skip-cameras > preload.log 2>&1 &
   ```

4. **定期检查日志**：
   ```bash
   tail -f preload.log
   ```
