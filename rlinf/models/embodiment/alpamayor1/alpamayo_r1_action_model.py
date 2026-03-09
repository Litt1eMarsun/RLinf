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
Alpamayo-R1 model wrapper for RL training.
Uses token decode mode (without diffusion) for GRPO compatibility.
"""
import os
import logging
from typing import Any, Dict

import einops
import torch
import torch.nn.functional as F
from transformers.generation.logits_process import LogitsProcessorList

from .alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from .alpamayo_r1.models.base_model import ReasoningVLA
from .helper import ConditionalTrajLogitsProcessor, create_message, get_processor

logger = logging.getLogger(__name__)


class AlpamayoR1ForRL(AlpamayoR1):
    """
    Alpamayo-R1 model wrapper for RL training with RLinf.
    
    Key features:
    - Uses token decode mode (VLM generation only, no diffusion)
    - Implements predict_action_batch interface
    - Returns logprobs for GRPO training
    """
    
    def __init__(self, config, pretrained_modules=None, original_vocab_size=None):
        # Bypass AlpamayoR1.__init__ to avoid instantiating action expert components
        # (self.expert, self.action_space, self.diffusion, self.action_in_proj,
        #  self.action_out_proj) that are only used for diffusion-based inference
        # and are not needed in RL token-decode mode.
        ReasoningVLA.__init__(self, config, pretrained_modules, original_vocab_size)
        self.post_init()

        # Initialize processor
        self.processor = get_processor(self.tokenizer)

        # Get trajectory token IDs for filtering
        self.traj_future_start_id = self.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
        self.traj_future_end_id = self.tokenizer.convert_tokens_to_ids("<|traj_future_end|>")

        logger.info(f"AlpamayoR1ForRL initialized with traj tokens: start={self.traj_future_start_id}, end={self.traj_future_end_id}")
    
    @property
    def language_model(self):
        """
        Alias for self.vlm to make it compatible with FSDP wrap policy detection.
        FSDP utils check for 'language_model' attribute to find transformer layers.
        """
        return self.vlm
    
    def forward(
        self,
        forward_inputs: Dict[str, torch.Tensor] = None,
        compute_logprobs: bool = False,
        compute_entropy: bool = False,
        compute_values: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass for training/evaluation.
        
        Args:
            forward_inputs: Dictionary containing recompute inputs from predict_action_batch
            compute_logprobs: Whether to compute log probabilities
            compute_entropy: Whether to compute entropy (not supported yet)
            compute_values: Whether to compute values (returns zeros as placeholder)
            use_cache: Whether to use KV cache (not used in this implementation)
            **kwargs: Additional arguments
        
        Returns:
            Dictionary containing:
                - logprobs: Token-level log probabilities [B, num_traj_tokens]
                - entropy: Placeholder None
                - values: Placeholder zeros [B, 1]
        """
        if forward_inputs is None:
            raise ValueError("forward_inputs is required for AlpamayoR1ForRL.forward()")
        
        device = self.vlm.device
        
        # import pdb; pdb.set_trace()
        # Extract inputs saved from predict_action_batch
        tokenized_data = {}
        generated_sequences = forward_inputs["generated_sequences"].to(device)  # Full sequence
        pl = forward_inputs["prompt_length"]
        prompt_length = pl[0] if isinstance(pl, list) else (int(pl.item()) if isinstance(pl, torch.Tensor) else pl)
        for k, v in forward_inputs.items():
            if k.startswith("tokenized_data_"):
                filename = k[len("tokenized_data_"):]
                tokenized_data[filename] = v.to(device)
        
        
        batch_size = generated_sequences.shape[0]
        
        # Fix attention_mask length mismatch:
        # tokenized_data["attention_mask"] was saved during predict_action_batch and only covers
        # the prompt tokens, but generated_sequences = prompt + generated tokens (longer).
        # Extend the mask with 1s for the newly generated tokens.
        if "attention_mask" in tokenized_data:
            gen_len = generated_sequences.shape[1]
            mask_len = tokenized_data["attention_mask"].shape[1]
            if gen_len > mask_len:
                extra_ones = torch.ones(tokenized_data["attention_mask"].shape[0],gen_len-mask_len,
                    dtype = tokenized_data["attention_mask"].dtype,
                    device = device)
                tokenized_data["attention_mask"] = torch.cat(
                    [tokenized_data["attention_mask"],extra_ones],dim=1
                )

        # Run teacher-forced forward pass through VLM
        # generated_sequences contains: prompt + generated trajectory tokens
        assert generated_sequences.max() < self.vlm.config.vocab_size

        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device)/1024**3
            reserved = torch.cuda.memory_reserved(device) /1024**3
            max_allocated = torch.cuda.max_memory_allocated(device)/1024**3
            print(f"[Before VLM forward] GPU {device} Memory:")
            print(f"  Allocated: {allocated:.2f} GB")
            print(f"  Reserved:  {reserved:.2f} GB")
            print(f"  Max Allocated: {max_allocated:.2f} GB")
            print(f"  Batch size: {batch_size}")
            print(f"  Input IDs shape: {generated_sequences.shape}")
        #import ipdb; ipdb.set_trace()
        torch.cuda.memory._record_memory_history(max_entries=100000)
        try:
            outputs = self.vlm( 
                input_ids=generated_sequences,
                **tokenized_data,
                use_cache=False,
                output_hidden_states=compute_values,
                return_dict=True,
            )
        except torch.cuda.OutOfMemoryError as e:
            snapshot_path = f"/tmp/oom_snapshot_rank{os.environ.get('LOCAL_RANK',0)}.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_path)
            print(f"[oom] memory snapshop save to {snapshot_path}")
            torch.cuda.memory._record_memory_history(enabled=None)
            raise


           # Print GPU memory after forward
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            print(f"[After VLM forward] GPU {device} Allocated: {allocated:.2f} GB")
        
        
        logits = outputs.logits  # [B, seq_len, vocab_size]
        
        # Extract trajectory token logits (after prompt, before end token)
        # For next-token prediction: logits[:, t] predicts token at position t+1
        # So logits[:, prompt_length-1:-1] predicts tokens at [prompt_length:]
        traj_token_logits = logits[:, prompt_length-1:-1, :]  # [B, gen_length, vocab_size]
        
        # Get generated token IDs (after prompt)
        generated_tokens = generated_sequences[:, prompt_length:]  # [B, gen_length]

        # Reuse ConditionalTrajLogitsProcessor to apply the same vocabulary mask
        # as during rollout, ensuring log_softmax normalises over the same token set.
        action_space_dims = self.traj_tokenizer.action_space.get_action_space_dims()
        tokens_per_traj = 1
        for d in action_space_dims:
            tokens_per_traj *= d
        processor = ConditionalTrajLogitsProcessor(
            traj_start_id=self.traj_future_start_id,
            traj_end_id=self.traj_future_end_id,
            traj_token_offset=self.future_token_start_idx,
            traj_vocab_size=self.traj_tokenizer.vocab_size,
            tokens_per_traj=tokens_per_traj,
        )
        gen_length = traj_token_logits.shape[1]
        for t in range(gen_length):
            processor(generated_sequences[:, :prompt_length + t], traj_token_logits[:, t, :])

        log_probs = F.log_softmax(traj_token_logits, dim=-1)
        
        # Filter to trajectory tokens only (between start and end markers)
        logprobs_list = []
        for b in range(batch_size):
            seq = generated_tokens[b]
            log_probs_seq = log_probs[b]
            
            # Find trajectory tokens between start and end markers
            start_mask = seq == self.traj_future_start_id
            end_mask = seq == self.traj_future_end_id
            
            if start_mask.any() and end_mask.any():
                start_pos = torch.where(start_mask)[0][0].item() + 1
                end_pos = torch.where(end_mask)[0][0].item()
            else:
                start_pos = 0
                end_pos = len(seq)
            
            # Extract logprobs for trajectory tokens
            traj_ids = seq[start_pos:end_pos]
            traj_log_probs_seq = log_probs_seq[start_pos:end_pos]
            
            # Gather logprobs of actual tokens
            token_log_probs = torch.gather(
                traj_log_probs_seq,
                dim=-1,
                index=traj_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            logprobs_list.append(token_log_probs)
        
        # Pad logprobs to same length
        max_traj_len = max(len(lp) for lp in logprobs_list) if logprobs_list else 1
        logprobs_padded = torch.zeros(
            (batch_size, max_traj_len),
            dtype=torch.float32,
            device=device
        )
        for b, lp in enumerate(logprobs_list):
            if len(lp) > 0:
                logprobs_padded[b, :len(lp)] = lp
        
        # Prepare outputs
        result = {
            "logprobs": logprobs_padded,
            "entropy": None if not compute_entropy else torch.zeros_like(logprobs_padded),
            "values": torch.zeros(batch_size, 1, device=device) if compute_values else None,
        }
        
        return result
    
    def predict_action_batch(self, batch: Dict[str, Any], mode: str = "train", **kwargs) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Predict actions for a batch of observations.
        
        Args:
            batch: Dictionary containing:
                - image_frames: [B, N_cam, N_frame, 3, H, W]
                - ego_history_xyz: [B, 1, T_hist, 3]
                - ego_history_rot: [B, 1, T_hist, 3, 3]
                - ego_future_xyz: [B, 1, T_fut, 3] (ground truth for reward)
                - ego_future_rot: [B, 1, T_fut, 3, 3] (ground truth)
            mode: "train" or "eval"
            **kwargs: Additional generation parameters
        
        Returns:
            actions: Flattened trajectory [B, action_dim]
            result: Dictionary with:
                - action: Same as actions
                - prev_logprobs: Token-level logprobs [B, num_traj_tokens]
                - prev_values: Placeholder zeros [B, 1]
                - forward_inputs: Input data for recompute
                - pred_xyz: Predicted positions [B, T_fut, 3]
                - pred_rot: Predicted rotations [B, T_fut, 3, 3]
                - gt_xyz: Ground truth positions [B, 1, T_fut, 3]
                - gt_rot: Ground truth rotations [B, 1, T_fut, 3, 3]
        """
        device = self.vlm.device
        batch_size = batch["image_frames"].shape[0]
        
        # 1. Prepare input - ensure all data is on the correct device
        image_frames = batch["image_frames"].to(device)  # [B, N_cam, N_frame, 3, H, W]
        ego_history_xyz = batch["ego_history_xyz"].to(device)  # [B, 1, T_hist, 3]
        ego_history_rot = batch["ego_history_rot"].to(device)  # [B, 1, T_hist, 3, 3]
        
        # Flatten cameras and frames: [B, N_cam, N_frame, 3, H, W] -> [B, N_cam*N_frame, 3, H, W]
        B, N_cam, N_frame = image_frames.shape[:3]
        image_frames_flat = einops.rearrange(image_frames, "b nc nf c h w -> b (nc nf) c h w")
        
        # Create messages for each batch item
        messages_list = []
        for b in range(batch_size):
            frames_b = image_frames_flat[b]  # [N_cam*N_frame, C, H, W]
            messages = create_message(frames_b)
            messages_list.append(messages)
        
        # 2. Process inputs with processor
        # Note: processor handles batching internally
        inputs = self.processor.apply_chat_template(
            messages_list,
            tokenize=True,
            add_generation_prompt=False,
            continue_final_message=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # Add trajectory history tokens
        tokenized_data = inputs
        input_ids = tokenized_data.pop("input_ids")
        traj_data_vlm = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data_vlm)
        
        # Move to device
        input_ids = input_ids.to(device)
        for key in tokenized_data:
            if isinstance(tokenized_data[key], torch.Tensor):
                tokenized_data[key] = tokenized_data[key].to(device)
        
        prompt_length = input_ids.shape[1] 
        # 3. Generate trajectory tokens with VLM
        #
        # tokens_per_traj: total AR trajectory tokens = product of action_space_dims
        # e.g. UnicycleAccelCurvature with 64 waypoints × 2 dims = 128 tokens
        action_space_dims = self.traj_tokenizer.action_space.get_action_space_dims()
        tokens_per_traj = 1
        for d in action_space_dims:
            tokens_per_traj *= d

        # max_new_tokens 必须足够容纳 CoT + 1(start) + tokens_per_traj + 1(end)
        # 默认给 CoT 留 256 token，加上轨迹部分
        default_max_new_tokens = 256 + 1 + tokens_per_traj + 1
        max_new_tokens = kwargs.get("max_new_tokens", default_max_new_tokens)

        # ConditionalTrajLogitsProcessor：
        #   CoT 阶段屏蔽轨迹 token；
        #   traj_future_start 之后限制只生成轨迹 token；
        #   tokens_per_traj 个轨迹 token 后强制输出 traj_future_end。
        traj_token_offset = self.future_token_start_idx
        traj_vocab_size = self.traj_tokenizer.vocab_size
        logits_processor = LogitsProcessorList([
            ConditionalTrajLogitsProcessor(
                traj_start_id=self.traj_future_start_id,
                traj_end_id=self.traj_future_end_id,
                traj_token_offset=traj_token_offset,
                traj_vocab_size=traj_vocab_size,
                tokens_per_traj=tokens_per_traj,
            )
        ])

        generation_config = self.vlm.generation_config
        generation_config.do_sample = kwargs.get("do_sample", True)
        generation_config.temperature = kwargs.get("temperature", 0.6)
        generation_config.top_p = kwargs.get("top_p", 0.98)
        generation_config.top_k = kwargs.get("top_k", None)
        generation_config.max_new_tokens = max_new_tokens
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.pad_token_id = self.tokenizer.pad_token_id

        with torch.no_grad():
            outputs = self.vlm.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                logits_processor=logits_processor,
                **tokenized_data,
            )
        
        # 4. Extract trajectory tokens
        generated_sequences = outputs.sequences  # [B, prompt_length + gen_length]
        generated_tokens = generated_sequences[:, prompt_length:]  # [B, gen_length]
        
        # Find trajectory tokens between <|traj_future_start|> and <|traj_future_end|>
        action_tokens_list = []
        for b in range(batch_size):
            seq = generated_tokens[b]
            
            # Find start and end positions
            start_mask = (seq == self.traj_future_start_id)
            end_mask = (seq == self.traj_future_end_id)
            
            if start_mask.any():
                start_pos = start_mask.nonzero(as_tuple=True)[0][0].item() + 1
            else:
                start_pos = 0
                logger.warning(f"No <|traj_future_start|> found in sequence {b}, using full sequence")
            
            if end_mask.any():
                end_pos = end_mask.nonzero(as_tuple=True)[0][0].item()
            else:
                end_pos = len(seq)
                logger.warning(f"No <|traj_future_end|> found in sequence {b}, using to end")
            
            action_tokens = seq[start_pos:end_pos]
            action_tokens_list.append(action_tokens)
        
        # Pad trajectory tokens to same length
        max_action_len = max(len(t) for t in action_tokens_list)
        action_tokens_padded = torch.full(
            (batch_size, max_action_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=device
        )
        for b, action_tokens in enumerate(action_tokens_list):
            action_tokens_padded[b, :len(action_tokens)] = action_tokens

        if action_tokens_padded.shape[1] != tokens_per_traj:
            if action_tokens_padded.shape[1] > tokens_per_traj:
                action_tokens_padded = action_tokens_padded[:, :tokens_per_traj]
            else:
                pad_len = tokens_per_traj - action_tokens_padded.shape[1]
                padding = torch.full(
                    (batch_size, pad_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                action_tokens_padded = torch.cat([action_tokens_padded, padding], dim=1)
        
        # 5. Decode trajectory using traj_tokenizer
        #
        # !! BUG FIX !!
        # action_tokens_padded 里存的是绝对词表 ID（traj_token_start_idx + relative_id）。
        # DiscreteTrajectoryTokenizer.decode() 期望的是 0-based 相对索引（0 ~ num_bins-1）。
        # 必须先减去偏移量，否则 action / (num_bins - 1) 会放大 ~150 倍，导致 1000m+ ADE。
        action_tokens_relative = (action_tokens_padded - self.future_token_start_idx).clamp(
            0, self.traj_tokenizer.vocab_size - 1
        )
        logger.debug(
            f"action_tokens_padded range: [{action_tokens_padded.min().item()}, "
            f"{action_tokens_padded.max().item()}]  "
            f"→ relative range: [{action_tokens_relative.min().item()}, "
            f"{action_tokens_relative.max().item()}]  "
            f"(traj_token_offset={self.future_token_start_idx})"
        )
        pred_xyz, pred_rot, _ = self.traj_tokenizer.decode(
            hist_xyz=ego_history_xyz[:, -1],  # [B, T_hist, 3]
            hist_rot=ego_history_rot[:, -1],  # [B, T_hist, 3, 3]
            tokens=action_tokens_relative,    # 0-based 相对索引
        )
        
        # 6. Compute logprobs from generation outputs
        logprobs = self._compute_token_logprobs(
            outputs.logits,
            generated_tokens,
            prompt_length
        )
        
        # Extract only trajectory token logprobs
        action_logprobs_list = []
        for b in range(batch_size):
            seq = generated_tokens[b]
            start_mask = (seq == self.traj_future_start_id)
            end_mask = (seq == self.traj_future_end_id)
            
            if start_mask.any():
                start_pos = start_mask.nonzero(as_tuple=True)[0][0].item() + 1
            else:
                start_pos = 0
            
            if end_mask.any():
                end_pos = end_mask.nonzero(as_tuple=True)[0][0].item()
            else:
                end_pos = len(seq)
            
            action_logprobs = logprobs[b, start_pos:end_pos]
            action_logprobs_list.append(action_logprobs)
        
        # Pad/truncate logprobs to tokens_per_traj (same as action_tokens)
        action_logprobs_padded = torch.zeros(
            (batch_size, tokens_per_traj),
            dtype=torch.float32,
            device=device
        )
        for b, action_logprobs in enumerate(action_logprobs_list):
            valid_len = min(len(action_logprobs), tokens_per_traj)
            action_logprobs_padded[b, :valid_len] = action_logprobs[:valid_len]
        
        # 7. Flatten trajectory to action vector
        # [B, T_fut, 3] + [B, T_fut, 3, 3] -> [B, T_fut*3 + T_fut*9]
        pred_xyz_flat = pred_xyz.flatten(1)  # [B, T_fut*3]
        pred_rot_flat = pred_rot.flatten(1)  # [B, T_fut*9]
        actions_traj = torch.cat([pred_xyz_flat, pred_rot_flat], dim=-1)  # [B, T_fut*12]
        
        # 8. Prepare result dictionary
        # Store all necessary inputs for recompute in forward()
        #import pdb; pdb.set_trace()

        forward_inputs = {
            "generated_sequences": generated_sequences,  # Full sequence: prompt + generated tokens
            "prompt_length": prompt_length,
            "image_frames": batch["image_frames"],
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        
        for k,v in tokenized_data.items():
            forward_inputs[f"tokenized_data_{k}"] = v 


        result = {
            "action": actions_traj,
            "prev_logprobs": action_logprobs_padded,
            "prev_values": torch.zeros(batch_size, 1, device=device),  # Placeholder
            "forward_inputs": forward_inputs,
            "pred_xyz": pred_xyz,  # [B, T_fut, 3]
            "pred_rot": pred_rot,  # [B, T_fut, 3, 3]
            "gt_xyz": batch.get("ego_future_xyz").to(device) if batch.get("ego_future_xyz") is not None else None,
            "gt_rot": batch.get("ego_future_rot").to(device) if batch.get("ego_future_rot") is not None else None,
        }
        
        return actions_traj, result
    
    def _compute_token_logprobs(
        self, 
        logits: tuple,
        generated_tokens: torch.Tensor,
        prompt_length: int
    ) -> torch.Tensor:
        """
        Compute token-level log probabilities from generation logits.
        
        Args:
            logits: Tuple of logits from generate(), each [B, vocab_size]
            generated_tokens: Generated token IDs [B, gen_length]
            prompt_length: Length of the prompt
        
        Returns:
            logprobs: Token-level log probabilities [B, gen_length]
        """
        batch_size, gen_length = generated_tokens.shape
        device = generated_tokens.device
        
        # Stack logits: list of [B, vocab_size] -> [B, gen_length, vocab_size]
        # HF generate() converts logits to float32 internally (_sample L2800).
        # Cast back to bfloat16 so log_softmax precision matches the FSDP bf16
        # forward path, keeping the policy ratio consistent (~1.0 for on-policy).
        logits_stacked = torch.stack(logits, dim=1).to(dtype=torch.bfloat16)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits_stacked, dim=-1)  # [B, gen_length, vocab_size]
        
        # Gather log probs of generated tokens
        token_log_probs = torch.gather(
            log_probs,
            dim=-1,
            index=generated_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [B, gen_length]
        
        return token_log_probs
    
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        """
        Enable gradient checkpointing for the VLM backbone.
        This forwards the call to the underlying VLM model (Qwen2VL).
        """
        if hasattr(self.vlm, 'gradient_checkpointing_enable'):
            logger.info("Enabling gradient checkpointing for VLM backbone")
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            logger.warning("VLM model does not support gradient_checkpointing_enable")
    
    def gradient_checkpointing_disable(self):
        """
        Disable gradient checkpointing for the VLM backbone.
        """
        if hasattr(self.vlm, 'gradient_checkpointing_disable'):
            logger.info("Disabling gradient checkpointing for VLM backbone")
            self.vlm.gradient_checkpointing_disable()
        else:
            logger.warning("VLM model does not support gradient_checkpointing_disable")
    
    @property
    def _no_split_modules(self):
        """Modules that should not be split across devices in FSDP.""" 
        # Qwen3VL uses Qwen2VL architecture internally
        return ["Qwen2VLDecoderLayer", "Qwen2VLAttention", "Qwen2DecoderLayer"]
    
    @_no_split_modules.setter
    def _no_split_modules(self, value):
        """Setter for _no_split_modules (required by transformers 5.0+)."""
        # Ignore the setter - we always return the same list
        pass