# 缓存迁移指南

## 问题说明

如果你之前运行过 preload_data.py，数据可能下载到了默认位置 `~/.cache/huggingface/` 而不是你期望的 `/data/workspace/RLinf/data/`。

## 已修复

脚本已更新，现在**保证**数据下载到指定的 `--cache-dir` 目录。

## 迁移选项

### 选项 1：移动现有数据（推荐，节省时间）

如果你已经下载了一些数据到 `~/.cache/huggingface/`，可以将其移动到目标位置：

```bash
# 检查现有缓存大小
du -sh ~/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/

# 创建目标目录
mkdir -p /data/workspace/RLinf/data/hub/

# 移动数据
mv ~/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/ \
   /data/workspace/RLinf/data/hub/

# 验证
ls -lh /data/workspace/RLinf/data/hub/
```

### 选项 2：重新下载（全新开始）

如果之前只下载了一点数据，可以删除旧数据，重新下载：

```bash
# 删除旧缓存
rm -rf ~/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/

# 清空跟踪文件（如果存在）
rm -f /data/workspace/RLinf/data/.downloaded_clips.txt
rm -f /home/sunyujie/workspace/RLinf/data/.downloaded_clips.txt

# 重新下载
cd /home/sunyujie/workspace/RLinf/examples/VA
python preload_data.py --skip-cached --skip-cameras
```

### 选项 3：创建符号链接（临时方案）

如果磁盘空间不足以移动数据：

```bash
# 创建符号链接
mkdir -p /data/workspace/RLinf/data/hub/
ln -s ~/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/ \
      /data/workspace/RLinf/data/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles
```

## 验证数据位置

运行以下命令验证数据确实在目标位置：

```bash
# 检查目标目录
ls -lh /data/workspace/RLinf/data/hub/

# 查看大小
du -sh /data/workspace/RLinf/data/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles/

# 测试访问速度（应该很快）
cd /data/workspace/RLinf
source .venv/bin/activate
python3 << 'EOF'
import os
os.environ['HF_HOME'] = '/data/workspace/RLinf/data/'
os.environ['HF_HUB_CACHE'] = '/data/workspace/RLinf/data/hub'

import physical_ai_av
avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

import time
clip_id = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
start = time.time()
egomotion = avdi.get_clip_feature(clip_id, avdi.features.LABELS.EGOMOTION, maybe_stream=True)
result = egomotion([5_100_000])
elapsed = time.time() - start

print(f"访问时间: {elapsed:.2f} 秒")
if elapsed < 1.0:
    print("✓ 数据已缓存在目标位置")
else:
    print("✗ 数据可能不在缓存中")
EOF
```

## 重要说明

### 为什么看不到 clip_id 文件夹？

这是**正常的**！HuggingFace Datasets 使用内部格式存储数据：

- 数据存储在 `blobs/` 目录中，使用哈希命名
- 不会创建以 clip_id 命名的文件夹
- 类似数据库的存储方式

### 如何确认数据已下载？

1. **查看目录大小**：
   ```bash
   du -sh /data/workspace/RLinf/data/hub/
   ```
   
2. **检查跟踪文件**：
   ```bash
   cat /data/workspace/RLinf/data/.downloaded_clips.txt
   ```

3. **测试访问速度**：已缓存的数据访问时间 < 1秒

## 现在开始下载

使用修复后的脚本下载数据：

```bash
cd /home/sunyujie/workspace/RLinf/examples/VA

# 推荐命令（智能跳过已下载 + 只下载 egomotion）
python preload_data.py --skip-cached --skip-cameras

# 下载完整数据（包含相机视频，很慢）
python preload_data.py --skip-cached
```

数据将确保下载到：`/data/workspace/RLinf/data/hub/`


