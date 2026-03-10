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
AVWorker for Alpamayo-R1 rollout with ReasoningRunner.
Handles batch data from AlpamayoAVDataset and computes trajectory rewards.
"""

import gc
import logging
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig

from rlinf.config import torch_dtype_from_precision
from rlinf.data.io_struct import RolloutRequest, RolloutResult
from rlinf.models import get_model
from rlinf.scheduler import Channel, CollectiveGroupOptions, Cluster, Worker
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.utils.utils import get_model_weights_id

logger = logging.getLogger(__name__)


class AVWorker(Worker):
    """
    Autonomous Vehicle rollout worker for Alpamayo-R1 model.
    
    This worker:
    1. Receives batches from AlpamayoAVDataset via channel
    2. Runs model inference to predict trajectories
    3. Computes rewards by comparing with ground truth (minADE)
    4. Returns rollout results for actor training
    """
    
    def __init__(self, cfg: DictConfig, placement: ModelParallelComponentPlacement):
        Worker.__init__(self)
        self._cfg = cfg
        self._placement = placement
        
        self._model = None
        self._return_logprobs = self._cfg.rollout.get("return_logprobs", True)
        
        # Actor communication settings
        self.actor_group_name = self._cfg.actor.group_name
        actor_world_size = self._placement.get_world_size("actor")
        self.actor_weight_src_rank = self._rank % actor_world_size
        
        # Weight tracking
        self.model_weights_id = ""
        self.count_update = 0
        
        # Sync weight communication options
        max_ctas = self._cfg.rollout.get("sync_weight_nccl_max_ctas", None)
        min_ctas = self._cfg.rollout.get("sync_weight_nccl_min_ctas", None)
        self._sync_weight_comm_options = CollectiveGroupOptions(
            accel_max_ctas=max_ctas, accel_min_ctas=min_ctas
        )
        
        # Sampling parameters
        self._sampling_params = self._get_sampling_params_from_config()
        
        logger.info(f"AVWorker initialized on rank {self._rank}")
    
    def _get_sampling_params_from_config(self) -> Dict[str, Any]:
        """Get sampling parameters from config."""
        cfg_sampling_params = self._cfg.algorithm.sampling_params
        
        sampling_params = {
            "do_sample": cfg_sampling_params.get("do_sample", True),
            "temperature": cfg_sampling_params.get("temperature_train", 0.6),
            "top_k": cfg_sampling_params.get("top_k", None),
            "top_p": cfg_sampling_params.get("top_p", 0.98),
            "max_new_tokens": cfg_sampling_params.get("max_new_tokens", 256),
        }
        
        return sampling_params
    
    async def init_worker(self) -> None:
        """Initialize the model."""
        logger.info(f"Initializing AVWorker on rank {self._rank}")
        
        self._enable_offload = self._cfg.rollout.get("enable_offload", False)
        self._device = torch.cuda.current_device()

        # Load model configuration
        rollout_model_config = self._cfg.rollout.model
        torch_dtype = torch_dtype_from_precision(rollout_model_config.precision)
        
        # Load model
        self._model = get_model(rollout_model_config)
        
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        
        self._model.eval()

        if self._enable_offload:
            self._offload_model()
            logger.info(f"AVWorker model offloaded to CPU after init (enable_offload=True)")
        
        logger.info(f"AVWorker initialized successfully on rank {self._rank}")

    def _offload_model(self) -> None:
        """将推理模型从 GPU 卸载到 CPU，释放显存供 Actor training 使用。"""
        self._model = self._model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"AVWorker model offloaded to CPU on rank {self._rank}")

    def _reload_model(self) -> None:
        """将推理模型从 CPU 重新加载到 GPU，准备 rollout。"""
        self._model = self._model.to(self._device)
        logger.info(f"AVWorker model reloaded to GPU on rank {self._rank}")
    
    async def sync_model_from_actor(self) -> None:
        """
        Sync model weights from actor worker.
        
        This method receives updated model parameters from the Actor worker
        after training and loads them into the rollout model for the next
        generation iteration.
        """
        logger.info(f"Syncing model weights from actor on rank {self._rank}")
        
        # Weight sync 需要模型在 GPU 上才能接收 NCCL 数据
        if self._enable_offload:
            self._reload_model()

        # Receive model state dict from actor
        param_state_dict = await self.recv(
            self.actor_group_name,
            src_rank=self.actor_weight_src_rank,
            async_op=True,
            options=self._sync_weight_comm_options,
        ).async_wait()
        
        # Load weights into model
        self._model.load_state_dict(param_state_dict)
        
        # Update weight tracking
        self.model_weights_id = (
            str(get_model_weights_id(self._model)) + f"_{self.count_update}"
        )
        self.count_update += 1
        
        # Clean up to free memory
        del param_state_dict
        gc.collect()
        torch.cuda.empty_cache()

        # sync 完成后立即 offload，让 Actor training 有足够显存
        if self._enable_offload:
            self._offload_model()
        
        logger.info(
            f"Model weights synced successfully on rank {self._rank}, "
            f"weights_id: {self.model_weights_id}"
        )
    
    async def rollout(self, input_channel: Channel, output_channel: Channel) -> None:
        """
        Main rollout loop.
        
        Args:
            input_channel: Channel to receive RolloutRequest from DataLoader
            output_channel: Channel to send RolloutResult to Actor
        """
        if self._enable_offload:
            self._reload_model()
        with self.worker_timer():
            await self._rollout_impl(input_channel, output_channel)
        if self._enable_offload:
            self._offload_model()
  
    def _expand_batch_for_group(self, batch: Dict[str, Any], group_size: int) -> Dict[str, Any]:
        """将 [B, ...] 的 batch 沿 batch 维复制 group_size 份，变成 [B*group_size, ...]。
        
        复制顺序为 scene-major（repeat_interleave），即：
        [s0, s0, ..., s0, s1, s1, ..., s1, ...]
        每个场景连续出现 group_size 次，保证同一场景的多次采样紧邻。
        """
        expanded = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                expanded[key] = value.repeat_interleave(group_size, dim=0)
            elif isinstance(value, list):
                expanded[key] = [item for item in value for _ in range(group_size)]
            else:
                expanded[key] = value
        return expanded

    async def _rollout_impl(self, input_channel: Channel, output_channel: Channel) -> None:
        logger.info(f"Starting rollout on rank {self._rank}")
        
        # Receive rollout request
        rollout_request: RolloutRequest = await input_channel.get(
            async_op=True
        ).async_wait()
        
        num_samples = len(rollout_request.multi_modal_inputs) if rollout_request.multi_modal_inputs else len(rollout_request.input_ids)
        logger.info(f"Received rollout request with {num_samples} samples")
        
        batch_data_list = rollout_request.multi_modal_inputs
        
        if batch_data_list is None or len(batch_data_list) == 0:
            logger.error("No batch data found in rollout request")
            return
        
        B = len(batch_data_list)
        group_size = self._cfg.algorithm.group_size
        device = self._model.vlm.device

        # Collate on CPU first, then expand (still on CPU), to avoid
        # holding group_size copies of the image tensor on GPU simultaneously.
        batch = self._collate_batch(batch_data_list)
        expanded_batch = self._expand_batch_for_group(batch, group_size)
        total_samples = B * group_size
        logger.info(f"Batch expanded: {B} scenes × {group_size} groups = {total_samples} samples")

        # 根据显存容量分 chunk 执行，避免一次性放不下
        rollout_micro_batch_size = self._cfg.rollout.get("micro_batch_size", total_samples)
        num_chunks = (total_samples + rollout_micro_batch_size - 1) // rollout_micro_batch_size
        
        chunk_results = []
        chunk_rewards = []

        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                start = chunk_idx * rollout_micro_batch_size
                end = min(start + rollout_micro_batch_size, total_samples)
                
                # Slice on CPU first, then move to GPU to keep peak GPU memory low.
                micro_batch_cpu = {}
                for key, value in expanded_batch.items():
                    if isinstance(value, (torch.Tensor, list)):
                        micro_batch_cpu[key] = value[start:end]
                    else:
                        micro_batch_cpu[key] = value
                micro_batch = self._move_to_device(micro_batch_cpu, device)

                actions, result = self._model.predict_action_batch(
                    micro_batch,
                    mode="train",
                    **self._sampling_params
                )
                rewards, selected_idx = self._compute_rewards(result)
                chunk_results.append(result)
                chunk_rewards.append(rewards)

        # 拼接所有 chunk 的结果: 已经是 [B*group_size, ...] scene-major 顺序
        if num_chunks == 1:
            merged_result = chunk_results[0]
            merged_rewards = chunk_rewards[0]
        else:
            merged_rewards = torch.cat(chunk_rewards, dim=0)
            merged_result = {}
            for key in chunk_results[0]:
                val = chunk_results[0][key]
                if isinstance(val, torch.Tensor):
                    merged_result[key] = torch.cat([r[key] for r in chunk_results], dim=0)
                elif isinstance(val, dict):
                    merged_dict = {}
                    for k, v in val.items():
                        if isinstance(v, torch.Tensor):
                            merged_dict[k] = torch.cat([r[key][k] for r in chunk_results], dim=0)
                        else:
                            merged_dict[k] = v
                    merged_result[key] = merged_dict
                else:
                    merged_result[key] = val

        # Prepare trajectory for actor
        trajectory = self._prepare_trajectory(
            rollout_request,
            merged_result,
            merged_rewards,
            batch_size=total_samples,
            selected_idx=selected_idx,
        )
        
        # Send trajectory to output channel
        await output_channel.put(trajectory, async_op=True).async_wait()
        
        logger.info(f"Rollout completed on rank {self._rank}")
    
    def _collate_batch(self, batch_data_list: list[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate list of batch items into a single batch.
        
        Args:
            batch_data_list: List of data dictionaries from AlpamayoAVDataset
        
        Returns:
            Collated batch dictionary
        """
        batch_size = len(batch_data_list)
        
        # Stack tensors
        batch = {}
        for key in batch_data_list[0].keys():
            if isinstance(batch_data_list[0][key], torch.Tensor):
                batch[key] = torch.stack([item[key] for item in batch_data_list], dim=0)
            else:
                batch[key] = [item[key] for item in batch_data_list]
        
        return batch
    
    def _move_to_device(self, batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
        """
        Move all tensors in batch to the specified device.
        
        Args:
            batch: Batch dictionary
            device: Target device
            
        Returns:
            Batch with all tensors on the target device
        """
        moved_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved_batch[key] = value.to(device)
            else:
                moved_batch[key] = value
        return moved_batch
    
    def _compute_rewards(self, result: Dict[str, Any], selected_idx: list = None):
        """
        Compute per-waypoint rewards based on trajectory prediction accuracy.
        
        Args:
            result: Model prediction result containing:
                - pred_xyz: [B, T_fut, 3]
                - gt_xyz: [B, 1, T_fut, 3]
            selected_idx: Waypoint indices to evaluate. If None, uses equal-spaced points.
        
        Returns:
            rewards: [B, n_waypoints] tensor of per-waypoint rewards
            selected_idx: list of selected waypoint indices
        """
        pred_xyz = result["pred_xyz"]  # [B, T_fut, 3]
        gt_xyz = result["gt_xyz"]  # [B, 1, T_fut, 3]
        
        if gt_xyz is None:
            logger.warning("No ground truth available, returning zero rewards")
            n_wp = len(selected_idx) if selected_idx else 4
            raise RuntimeError
        
        gt_xyz = gt_xyz[:, 0, :, :]  # [B, T_fut, 3]
        
        if selected_idx is None:
            pred_length = pred_xyz.shape[1]
            selected_gap = pred_length // 4
            selected_idx = [(i + 1) * selected_gap - 1 for i in range(4)]
        
        pred_xyz_select = pred_xyz[:, selected_idx, :]
        gt_xyz_select = gt_xyz[:, selected_idx, :]

        displacement = torch.norm(pred_xyz_select - gt_xyz_select, dim=-1)  # [B, n_waypoints]
        rewards = -displacement  # [B, n_waypoints]

        reward_type = self._cfg.algorithm.get("reward_type", "sequence_level")
        if reward_type != "waypoint_level":
            rewards = rewards.mean(dim=-1, keepdim=True)  # [B, 1]
            selected_idx = None
        
        logger.info(f"Computed rewards (reward_type={reward_type}) - mean: {rewards.mean().item():.4f}, "
                   f"std: {rewards.std(correction=0).item():.4f}, "
                   f"min: {rewards.min().item():.4f}, "
                   f"max: {rewards.max().item():.4f}")
        
        return rewards, selected_idx
    
    def _prepare_trajectory(
        self,
        rollout_request: RolloutRequest,
        result: Dict[str, Any],
        rewards: torch.Tensor,
        batch_size: int = None,
        selected_idx: list = None,
    ):
        """
        Prepare Trajectory object for actor training.
        
        Args:
            rollout_request: Original request
            result: Model prediction result
            rewards: Computed rewards [B, n_waypoints]
            batch_size: Total number of sequences (scenes * group_size).
            selected_idx: List of waypoint indices used for reward computation.
        
        Returns:
            Trajectory object
        """
        from rlinf.data.embodied_io_struct import Trajectory
        
        if batch_size is None:
            batch_size = len(rollout_request.multi_modal_inputs)
        
        prev_logprobs = result.get("prev_logprobs")  # [B, tokens_per_traj]
        pred_xyz = result.get("pred_xyz")  # [B, T_fut, 3]
        forward_inputs = result.get("forward_inputs")

        # Store selected_idx as a Python list in forward_inputs for pipeline transfer
        if selected_idx is not None:
            forward_inputs["selected_idx"] = selected_idx
        
        actions_flat = pred_xyz.reshape(batch_size, -1)  # [B, T_fut*3]
        actions = actions_flat.unsqueeze(0)  # [1, B, T_fut*3]
        
        # chunk_size = n_waypoints (rewards shape: [B, n_wp])
        n_wp = rewards.shape[-1] if rewards.dim() > 1 else 1
        
        # dones/terminations/truncations: [num_chunk+1, B, chunk_size] = [2, B, n_wp]
        # After preprocess flattening, dones become [n_steps+1, B].
        # Only the LAST waypoint marks episode end; intermediate waypoints are NOT done.
        # dones[0,:,:] = False (initial state before all waypoints)
        # dones[1,:,:-1] = False (intermediate waypoints continue)
        # dones[1,:,-1] = True (episode ends after last waypoint)
        dones_tensor = torch.zeros(2, batch_size, n_wp, dtype=torch.bool)
        dones_tensor[1, :, -1] = True
        terminations_tensor = torch.zeros(2, batch_size, n_wp, dtype=torch.bool)
        truncations_tensor = torch.zeros(2, batch_size, n_wp, dtype=torch.bool)
        
        trajectory = Trajectory(
            model_weights_id=f"{self.model_weights_id}",
            actions=actions,  # [1, B, T_fut*3]
            rewards=rewards.unsqueeze(0),  # [1, B, n_wp] - [num_chunk, bsz, chunk_size]
            prev_logprobs=prev_logprobs.unsqueeze(0) if prev_logprobs is not None else None,  # [1, B, tokens_per_traj]
            prev_values=None,
            dones=dones_tensor,  # [2, B, n_wp]
            terminations=terminations_tensor,
            truncations=truncations_tensor,
            intervene_flags=torch.zeros(1, batch_size, n_wp, dtype=torch.bool),
            forward_inputs=forward_inputs,
            max_episode_length=1,
        )
        
        return trajectory
    
    def _prepare_rollout_result(
        self,
        rollout_request: RolloutRequest,
        result: Dict[str, Any],
        rewards: torch.Tensor
    ) -> RolloutResult:
        """
        Prepare RolloutResult for actor training.
        
        Args:
            rollout_request: Original request
            result: Model prediction result
            rewards: Computed rewards
        
        Returns:
            RolloutResult object
        """
        batch_size = len(rollout_request.input_ids)
        group_size = rollout_request.n
        
        # Extract logprobs
        prev_logprobs = result["prev_logprobs"]  # [B, num_tokens]
        
        # Convert logprobs to list format for RolloutResult
        if self._return_logprobs:
            rollout_logprobs = []
            for b in range(batch_size):
                logprobs_b = prev_logprobs[b].cpu().tolist()
                rollout_logprobs.append(logprobs_b)
        else:
            rollout_logprobs = None
        
        # Prepare response data
        # For trajectory prediction, we treat the trajectory tokens as "response"
        response_ids = []
        response_lengths = []
        response_texts = []
        
        for b in range(batch_size):
            # Get trajectory representation as token IDs
            # We use dummy token IDs since we're not doing text generation
            num_tokens = prev_logprobs.shape[1]
            response_ids.append(list(range(num_tokens)))
            response_lengths.append(num_tokens)
            response_texts.append(f"trajectory_{b}")
        
        # Create RolloutResult
        rollout_result = RolloutResult(
            num_sequence=batch_size,
            group_size=group_size,
            prompt_lengths=[0] * batch_size,  # No prompt for trajectory task
            prompt_ids=[[]] * batch_size,
            response_lengths=response_lengths,
            response_ids=response_ids,
            is_end=[True] * batch_size,
            rewards=rewards.cpu().tolist(),
            advantages=None,  # Will be computed by actor
            prompt_texts=[""] * batch_size,
            response_texts=response_texts,
            answers=rollout_request.answers,
            image_data=rollout_request.image_data,
            multi_modal_inputs=rollout_request.multi_modal_inputs,
            response_mask=None,
            rollout_logprobs=rollout_logprobs,
            recompute_prev_logprobs=None,
            prev_logprobs=prev_logprobs.cpu() if self._return_logprobs else None,
            ref_logprobs=None,
        )
        
        return rollout_result
    
