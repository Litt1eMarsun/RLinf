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

"""Alpamayo-R1 model for RLinf."""

import sys

import torch
from omegaconf import DictConfig

# Import alpamayo_r1 submodule first to ensure it's loaded
from . import alpamayo_r1

# Register alpamayo_r1 module alias for Hydra instantiation
# This allows configs to reference "alpamayo_r1.action_space..." instead of the full path
# transformers的config需要alpamayo这个包，在缓存字典中直接创建
sys.modules["alpamayo_r1"] = sys.modules["rlinf.models.embodiment.alpamayor1.alpamayo_r1"]

from .alpamayo_r1_action_model import AlpamayoR1ForRL


def get_model(cfg: DictConfig, torch_dtype=None):
    """
    Load Alpamayo-R1 model for RL training.
    
    Args:
        cfg: Model configuration
        torch_dtype: Data type for model (e.g., torch.bfloat16)
    
    Returns:
        AlpamayoR1ForRL model instance
    """
    if torch_dtype is None:
        torch_dtype = torch.bfloat16
    
    model_path = cfg.get("model_path", "nvidia/Alpamayo-R1-10B")
    
    # Load model with RLinf wrapper
    model = AlpamayoR1ForRL.from_pretrained(
        model_path,
        dtype=torch_dtype
    )
    
    return model
