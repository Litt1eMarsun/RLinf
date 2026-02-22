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
        
        # Load model configuration
        rollout_model_config = self._cfg.rollout.model
        torch_dtype = torch_dtype_from_precision(rollout_model_config.precision)
        
        # Load model
        self._model = get_model(rollout_model_config)
        
        if torch.cuda.is_available():
            self._model = self._model.cuda()
        
        self._model.eval()
        
        logger.info(f"AVWorker initialized successfully on rank {self._rank}")
    
    async def sync_model_from_actor(self) -> None:
        """
        Sync model weights from actor worker.
        
        This method receives updated model parameters from the Actor worker
        after training and loads them into the rollout model for the next
        generation iteration.
        """
        logger.info(f"Syncing model weights from actor on rank {self._rank}")
        
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
        with self.worker_timer():
            await self._rollout_impl(input_channel, output_channel)

    async def _rollout_impl(self, input_channel: Channel, output_channel: Channel) -> None:
        logger.info(f"Starting rollout on rank {self._rank}")
        
        # Receive rollout request
        rollout_request: RolloutRequest = await input_channel.get(
            async_op=True
        ).async_wait()
        
        num_samples = len(rollout_request.multi_modal_inputs) if rollout_request.multi_modal_inputs else len(rollout_request.input_ids)
        logger.info(f"Received rollout request with {num_samples} samples")
        
        # Process the batch
        # For AV tasks, data is in multi_modal_inputs, not input_ids
        batch_data_list = rollout_request.multi_modal_inputs
        
        if batch_data_list is None or len(batch_data_list) == 0:
            logger.error("No batch data found in rollout request")
            return
        
        # Collate batch data
        batch = self._collate_batch(batch_data_list)
        device = self._model.vlm.device
        batch = self._move_to_device(batch, device)
        
        # Run model inference
        with torch.no_grad():
            actions, result = self._model.predict_action_batch(
                batch,
                mode="train",
                **self._sampling_params
            )
        
        # Compute rewards
        rewards = self._compute_rewards(result)
        
        # Prepare trajectory for actor
        trajectory = self._prepare_trajectory(
            rollout_request,
            result,
            rewards
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
    
    def _compute_rewards(self, result: Dict[str, Any]) -> torch.Tensor:
        """
        Compute rewards based on trajectory prediction accuracy.
        Uses minimum Average Displacement Error (minADE).
        
        Args:
            result: Model prediction result containing:
                - pred_xyz: [B, T_fut, 3]
                - gt_xyz: [B, 1, T_fut, 3]
        
        Returns:
            rewards: [B] tensor of rewards
        """
        pred_xyz = result["pred_xyz"]  # [B, T_fut, 3]
        gt_xyz = result["gt_xyz"]  # [B, 1, T_fut, 3]
        
        if gt_xyz is None:
            logger.warning("No ground truth available, returning zero rewards")
            return torch.zeros(pred_xyz.shape[0], device=pred_xyz.device)
        
        # Remove the trajectory group dimension from ground truth
        gt_xyz = gt_xyz[:, 0, :, :]  # [B, T_fut, 3]
        
        # Compute ADE (Average Displacement Error)
        # ADE = mean of L2 distances across time steps
        displacement = torch.norm(pred_xyz - gt_xyz, dim=-1)  # [B, T_fut]
        ade = displacement.mean(dim=-1)  # [B]
        
        # Convert to reward (negative error)
        # We use negative ADE as reward (higher is better)
        rewards = -ade
        
        # Optional: Apply reward shaping (e.g., exponential)
        # rewards = torch.exp(-ade / threshold)
        
        logger.info(f"Computed rewards - mean: {rewards.mean().item():.4f}, "
                   f"std: {rewards.std().item():.4f}, "
                   f"min: {rewards.min().item():.4f}, "
                   f"max: {rewards.max().item():.4f}")
        
        return rewards
    
    def _prepare_trajectory(
        self,
        rollout_request: RolloutRequest,
        result: Dict[str, Any],
        rewards: torch.Tensor
    ):
        """
        Prepare Trajectory object for actor training.
        
        Args:
            rollout_request: Original request
            result: Model prediction result
            rewards: Computed rewards
        
        Returns:
            Trajectory object
        """
        from rlinf.data.embodied_io_struct import Trajectory
        
        batch_size = len(rollout_request.multi_modal_inputs)
        
        # Extract data from result
        prev_logprobs = result.get("prev_logprobs")  # [B, num_tokens]
        pred_xyz = result.get("pred_xyz")  # [B, T_fut, 3] - predicted trajectories
        forward_inputs = result.get("forward_inputs")
        # Create Trajectory object
        # Note: Trajectory expects [T, B, ...] format for all tensor fields
        # pred_xyz is [B, T_fut, 3], we need to reshape it to [1, B, T_fut*3]
        # to match the single-step embodied format
        
        # Flatten trajectory: [B, T_fut, 3] -> [B, T_fut*3]
        actions_flat = pred_xyz.reshape(batch_size, -1)  # [B, T_fut*3]
        actions = actions_flat.unsqueeze(0)  # [1, B, T_fut*3] - single timestep
        
        # Embodied RL framework expects dones/terminations/truncations to have
        # num_chunk + 1 time steps (T+1), while rewards/actions have num_chunk (T).
        # For single-step AV prediction: num_chunk=1, so dones needs [2, B, 1].
        # First row = initial state (not done), second row = after prediction (done).
        dones_tensor = torch.zeros(2, batch_size, 1, dtype=torch.bool)
        dones_tensor[1] = True  # Episode ends after the single prediction step
        terminations_tensor = torch.zeros(2, batch_size, 1, dtype=torch.bool)
        truncations_tensor = torch.zeros(2, batch_size, 1, dtype=torch.bool)
        
        trajectory = Trajectory(
            model_weights_id=f"{self.model_weights_id}",
            actions=actions,  # [1, B, T_fut*3] - flattened trajectory
            rewards=rewards.unsqueeze(0).unsqueeze(-1),  # [1, B, 1] - [num_chunk, bsz, chunk_size]
            prev_logprobs=prev_logprobs.unsqueeze(0) if prev_logprobs is not None else None,  # [1, B, num_tokens]
            prev_values=None,  # No critic for GRPO
            dones=dones_tensor,  # [2, B, 1] - [num_chunk+1, bsz, chunk_size]
            terminations=terminations_tensor,  # [2, B, 1]
            truncations=truncations_tensor,  # [2, B, 1]
            intervene_flags=torch.zeros(1, batch_size, 1, dtype=torch.bool),  # [1, B, 1]
            forward_inputs=forward_inputs,  # Only simple tensors
            max_episode_length=1,  # Single-step prediction
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
    
