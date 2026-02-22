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

"""Dataset loader for PhysicalAI-AV (Alpamayo) autonomous driving data."""

from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import physical_ai_av
import scipy.spatial.transform as spt
import torch
from einops import rearrange
from omegaconf import DictConfig
from torch.utils.data import Dataset


class AlpamayoAVDataset(Dataset):
    """
    PyTorch Dataset for PhysicalAI-AV autonomous vehicle dataset.
    
    This dataset loads multi-camera video frames and egomotion trajectories
    from the PhysicalAI-AV dataset for autonomous driving tasks.
    
    Features:
    - Multi-camera video frames (1920x1080 resolution)
    - Historical ego-vehicle trajectories (position + rotation)
    - Future ego-vehicle trajectories for prediction
    - Temporal alignment of camera frames and trajectories
    
    Args:
        config: Configuration dict/object with the following keys:
            - data_path: Path to parquet file containing clip IDs
            - num_history_steps: Number of history trajectory steps (default: 16)
            - num_future_steps: Number of future trajectory steps (default: 64)
            - time_step: Time step in seconds (default: 0.1 for 10Hz)
            - num_frames: Number of frames per camera (default: 4)
            - camera_features: Optional list of camera names to use
            - maybe_stream: Whether to stream from HuggingFace (default: True)
            - initial_timestamp_us: Initial timestamp in microseconds (default: 5_100_000)
        
    Returns:
        Dictionary with keys:
            - image_frames: (N_cameras, num_frames, 3, H, W)
            - camera_indices: (N_cameras,)
            - ego_history_xyz: (1, 1, num_history_steps, 3)
            - ego_history_rot: (1, 1, num_history_steps, 3, 3)
            - ego_future_xyz: (1, 1, num_future_steps, 3)
            - ego_future_rot: (1, 1, num_future_steps, 3, 3)
            - relative_timestamps: (N_cameras, num_frames)
            - absolute_timestamps: (N_cameras, num_frames)
            - clip_id: str
            - t0_us: int
    """
    
    def __init__(
        self,
        config: Union[DictConfig, dict],
    ):
        super().__init__()
        self.config = config
        
        # Initialize PhysicalAI-AV dataset interface
        # Use cache_dir from config if provided, otherwise use default
        cache_dir = config.get("cache_dir", None)
        self.avdi = physical_ai_av.PhysicalAIAVDatasetInterface(cache_dir=cache_dir)
        
        # Load clip IDs from parquet file
        self.clip_ids = config.get("clip_ids", None)
        if self.clip_ids is None:
            ## 在这里加载clip_ids.parque
            data_path = config.get("data_path") or config.get("data_paths")
            if data_path is None:
                raise ValueError("Must provide 'data_path' or 'data_paths' in config")
            self.clip_ids = self._load_clip_ids(data_path)
        
        # Dataset parameters
        self.num_history_steps = config.get("num_history_steps", 16)
        self.num_future_steps = config.get("num_future_steps", 64)
        self.time_step = config.get("time_step", 0.1)  # 10Hz
        self.num_frames = config.get("num_frames", 4)
        self.maybe_stream = config.get("maybe_stream", True)
        self.initial_timestamp_us = config.get("initial_timestamp_us", 5_100_000)
        
        # Camera configuration
        camera_feature_names = config.get("camera_features", None)
        if camera_feature_names is None:
            # Default: 4 cameras
            self.camera_features = [
                self.avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
                self.avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
                self.avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
                self.avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
            ]
        else:
            self.camera_features = [
                getattr(self.avdi.features.CAMERA, name.upper())
                for name in camera_feature_names
            ]
        
        # Camera name to index mapping
        self.camera_name_to_index = {
            "camera_cross_left_120fov": 0,
            "camera_front_wide_120fov": 1,
            "camera_cross_right_120fov": 2,
            "camera_rear_left_70fov": 3,
            "camera_rear_tele_30fov": 4,
            "camera_rear_right_70fov": 5,
            "camera_front_tele_30fov": 6,
        }
        
    def _load_clip_ids(self, data_path: str) -> list[str]:
        """Load clip IDs from parquet file."""
        print(f"load clip ids from {data_path}...")
        import time
        start_time = time.time()
        df = pd.read_parquet(data_path)
        df_reset = df.reset_index()
        train_clips = df_reset[df_reset['split'] == 'train']['clip_id'].tolist()
        print(f"load clip ids done, time: {time.time() - start_time:.2f}s")
        return train_clips[:64]
    
    def __len__(self) -> int:
        return len(self.clip_ids)
    
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """
        Load a single sample from the dataset.
        
        Args:
            idx: Index of the sample (corresponds to clip_id index)
            
        Returns:
            Dictionary containing multi-camera images and trajectory data
        """
        import time
        t_start = time.time()
        
        clip_id = self.clip_ids[idx]
        t0_us = self.initial_timestamp_us
        
        # Load egomotion trajectories
        t0 = time.time()
        print(f"load egomotion...")
        ego_data = self._load_egomotion(clip_id, t0_us)
        t_ego = time.time() - t0
        print(f"load egomotion done, time: {t_ego:.2f}s")
        
        # Load multi-camera video frames
        t0 = time.time()
        print(f"load video frames...")
        video_data = self._load_video_frames(clip_id, t0_us)
        t_video = time.time() - t0
        print(f"load video frames done, time: {t_video:.2f}s")
        
        t_total = time.time() - t_start
        print(f"[Dataset] sample {idx}: ego={t_ego:.2f}s, video={t_video:.2f}s, total={t_total:.2f}s")
        
        # Combine all data
        return {
            **video_data,
            **ego_data,
            "clip_id": clip_id,
            "t0_us": t0_us,
        }
    
    def _load_egomotion(self, clip_id: str, t0_us: int) -> dict[str, torch.Tensor]:
        """Load and process egomotion trajectories."""
        # Load egomotion data
        egomotion = self.avdi.get_clip_feature(
            clip_id,
            self.avdi.features.LABELS.EGOMOTION,
            maybe_stream=False,
        )
        
        # Validate timestamp
        min_timestamp = self.num_history_steps * self.time_step * 1_000_000
        if t0_us <= min_timestamp:
            raise ValueError(
                f"t0_us ({t0_us}) must be greater than history time range ({min_timestamp})"
            )
        
        # Compute timestamps for trajectory sampling
        # History: [..., t0-0.2s, t0-0.1s, t0] (num_history_steps points ending at t0)
        history_offsets_us = np.arange(
            -(self.num_history_steps - 1) * self.time_step * 1_000_000,
            self.time_step * 1_000_000 / 2,
            self.time_step * 1_000_000,
        ).astype(np.int64)
        history_timestamps = t0_us + history_offsets_us
        
        # Future: [t0+0.1s, t0+0.2s, ..., t0+6.4s] (num_future_steps points after t0)
        future_offsets_us = np.arange(
            self.time_step * 1_000_000,
            (self.num_future_steps + 0.5) * self.time_step * 1_000_000,
            self.time_step * 1_000_000,
        ).astype(np.int64)
        future_timestamps = t0_us + future_offsets_us
        
        # Get egomotion at history and future timestamps
        ego_history = egomotion(history_timestamps)
        ego_history_xyz = ego_history.pose.translation  # (num_history_steps, 3)
        ego_history_quat = ego_history.pose.rotation.as_quat()  # (num_history_steps, 4)
        
        ego_future = egomotion(future_timestamps)
        ego_future_xyz = ego_future.pose.translation  # (num_future_steps, 3)
        ego_future_quat = ego_future.pose.rotation.as_quat()  # (num_future_steps, 4)
        
        # Transform to local frame (relative to t0 pose)
        # The model expects trajectories in the ego frame at t0.
        t0_xyz = ego_history_xyz[-1].copy()  # Position at t0
        t0_quat = ego_history_quat[-1].copy()  # Orientation at t0
        t0_rot = spt.Rotation.from_quat(t0_quat)
        t0_rot_inv = t0_rot.inv()
        
        # Transform history positions to local frame
        ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)
        
        # Transform future positions to local frame
        ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)
        
        # Transform rotations to local frame
        ego_history_rot_local = (
            t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)
        ).as_matrix()
        ego_future_rot_local = (
            t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)
        ).as_matrix()
        
        # Convert to torch tensors with n_traj dimension: (n_traj=1, T, ...)
        # Batch dimension will be added by collate_fn during batching
        return {
            "ego_history_xyz": torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0),  # [1, T, 3]
            "ego_history_rot": torch.from_numpy(ego_history_rot_local).float().unsqueeze(0),  # [1, T, 3, 3]
            "ego_future_xyz": torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0),    # [1, T, 3]
            "ego_future_rot": torch.from_numpy(ego_future_rot_local).float().unsqueeze(0),    # [1, T, 3, 3]
        }
    
    def _load_video_frames(self, clip_id: str, t0_us: int) -> dict[str, torch.Tensor]:
        """Load multi-camera video frames."""
        image_frames_list = []
        camera_indices_list = []
        timestamps_list = []
        
        # Image timestamps: if num_frames=4, load at [t0-0.3s, t0-0.2s, t0-0.1s, t0]
        image_timestamps = np.array(
            [
                t0_us - (self.num_frames - 1 - i) * int(self.time_step * 1_000_000)
                for i in range(self.num_frames)
            ],
            dtype=np.int64,
        )
        
        for cam_feature in self.camera_features:
            # Get camera data
            camera = self.avdi.get_clip_feature(
                clip_id,
                cam_feature,
                maybe_stream=False,
            )
            
            # Decode video frames: (num_frames, H=1080, W=1920, C=3) uint8
            frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)
            
            # Convert to (num_frames, C, H, W) for model input
            frames_tensor = torch.from_numpy(frames)
            frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")
            
            # Extract camera name and index
            if isinstance(cam_feature, str):
                cam_name = cam_feature.split("/")[-1] if "/" in cam_feature else cam_feature
                cam_name = cam_name.lower()
            else:
                raise ValueError(f"Unexpected camera feature type: {type(cam_feature)}")
            cam_idx = self.camera_name_to_index.get(cam_name, 0)
            
            image_frames_list.append(frames_tensor)
            camera_indices_list.append(cam_idx)
            timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))
        
        # Stack all cameras: (N_cameras, num_frames, 3, H, W)
        image_frames = torch.stack(image_frames_list, dim=0)
        camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)
        all_timestamps = torch.stack(timestamps_list, dim=0)
        
        # Sort by camera index for consistent ordering
        sort_order = torch.argsort(camera_indices)
        image_frames = image_frames[sort_order]
        camera_indices = camera_indices[sort_order]
        all_timestamps = all_timestamps[sort_order]
        
        # Compute relative timestamps in seconds
        camera_tmin = all_timestamps.min()
        relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6
        
        return {
            "image_frames": image_frames,  # (N_cameras, num_frames, 3, H, W)
            "camera_indices": camera_indices,  # (N_cameras,)
            "relative_timestamps": relative_timestamps,  # (N_cameras, num_frames)
            "absolute_timestamps": all_timestamps,  # (N_cameras, num_frames)
        }


def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int = 5_100_000,
    avdi: Optional[physical_ai_av.PhysicalAIAVDatasetInterface] = None,
    maybe_stream: bool = True,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: Optional[list] = None,
    num_frames: int = 4,
) -> dict[str, Any]:
    """
    Legacy function for loading PhysicalAI-AV data (backward compatibility).
    
    This function provides the same interface as the original load_physical_aiavdataset
    but now uses the AlpamayoAVDataset class internally.
    
    For new code, prefer using AlpamayoAVDataset directly.
    
    Args:
        clip_id: The clip ID to load data from
        t0_us: The timestamp (in microseconds) at which to sample
        avdi: Optional pre-initialized PhysicalAIAVDatasetInterface
        maybe_stream: Whether to stream data from HuggingFace
        num_history_steps: Number of history trajectory steps (default: 16)
        num_future_steps: Number of future trajectory steps (default: 64)
        time_step: Time step between trajectory points in seconds (default: 0.1)
        camera_features: List of camera features to load
        num_frames: Number of frames per camera to load (default: 4)
    
    Returns:
        Dictionary with image frames and trajectory data
    """
    # Create a temporary config
    config = {
        "data_path": None,  # Not used for single-sample loading
        "num_history_steps": num_history_steps,
        "num_future_steps": num_future_steps,
        "time_step": time_step,
        "num_frames": num_frames,
        "maybe_stream": maybe_stream,
        "initial_timestamp_us": t0_us,
    }
    
    # Create dataset instance (temporarily inject clip_id)
    dataset = AlpamayoAVDataset(config)
    if avdi is not None:
        dataset.avdi = avdi
    if camera_features is not None:
        dataset.camera_features = camera_features
    
    # Override clip_ids with single clip
    dataset.clip_ids = [clip_id]
    
    # Load the data
    return dataset[0]
