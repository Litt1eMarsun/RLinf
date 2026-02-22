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
Test script for Alpamayo-R1 rollout.
Tests model inference and reward computation without full training.
"""

import torch
from omegaconf import OmegaConf

from rlinf.config import torch_dtype_from_precision
from rlinf.data.datasets.alpamayo_av import AlpamayoAVDataset
from rlinf.models import get_model


def test_dataset():
    """Test dataset loading."""
    print("=" * 80)
    print("Testing AlpamayoAVDataset...")
    print("=" * 80)
    
    config = OmegaConf.create({
        "data_path": "path/to/vla_golden_train.parquet",  # Update this
        "clip_ids": ["030c760c-ae38-49aa-9ad8-f5650a545d26"],
        "num_history_steps": 16,
        "num_future_steps": 64,
        "time_step": 0.1,
        "num_frames": 4,
        "maybe_stream": True,
        "initial_timestamp_us": 5100000,
    })
    
    dataset = AlpamayoAVDataset(config)
    print(f"✓ Dataset object initialized: {len(dataset)} clips available")
    print(f"  Note: Actual data downloads when accessing dataset[0]")
    print(f"\nAttempting to load first sample...")
    print(f"  WARNING: This may take 5-15 minutes for first-time download from HuggingFace")
    print(f"  - Downloading EGOMOTION data: ~5 min")
    print(f"  - Downloading 4 camera videos: ~10 min")
    
    import time
    start_time = time.time()
        
        # Try to load one sample
    sample = dataset[0]
    
    elapsed = time.time() - start_time
    print(f"\n✓ Sample loaded successfully (took {elapsed:.1f} seconds)")
    print(f"  - image_frames shape: {sample['image_frames'].shape}")
    print(f"  - ego_history_xyz shape: {sample['ego_history_xyz'].shape}")
    print(f"  - ego_history_rot shape: {sample['ego_history_rot'].shape}")
    print(f"  - ego_future_xyz shape: {sample['ego_future_xyz'].shape}")
    print(f"  - ego_future_rot shape: {sample['ego_future_rot'].shape}")


def test_model_loading():
    """Test model loading."""
    print("\n" + "=" * 80)
    print("Testing Model Loading...")
    print("=" * 80)
    
    config = OmegaConf.create({
        "model_type": "alpamayo",
        "model_path": "nvidia/Alpamayo-R1-10B",
        "precision": "bf16",
        "is_lora": False,
    })
    
    try:
        model = get_model(config)
        if torch.cuda.is_available():
            model = model.cuda()
        
        print(f"✓ Model loaded successfully")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Device: {next(model.parameters()).device}")
        print(f"  - Dtype: {next(model.parameters()).dtype}")
        
        return model
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_model_inference(model):
    """Test model inference on dummy data."""
    print("\n" + "=" * 80)
    print("Testing Model Inference...")
    print("=" * 80)
    
    try:
        # Create dummy batch
        batch_size = 2
        n_cam = 4
        n_frame = 4
        h, w = 1080, 1920
        t_hist = 16
        t_fut = 64
        
        device = next(model.parameters()).device
        
        batch = {
            "image_frames": torch.randint(0, 255, (batch_size, n_cam, n_frame, 3, h, w), dtype=torch.uint8).to(device),
            "ego_history_xyz": torch.randn(batch_size, 1, t_hist, 3).to(device),
            "ego_history_rot": torch.randn(batch_size, 1, t_hist, 3, 3).to(device),
            "ego_future_xyz": torch.randn(batch_size, 1, t_fut, 3).to(device),
            "ego_future_rot": torch.randn(batch_size, 1, t_fut, 3, 3).to(device),
        }
        
        print(f"Running inference on batch of {batch_size} samples...")
        
        with torch.no_grad():
            actions, result = model.predict_action_batch(
                batch,
                mode="train",
                do_sample=True,
                temperature=0.6,
                top_p=0.98,
                max_new_tokens=256,
            )
        
        print(f"✓ Inference completed successfully")
        print(f"  - Actions shape: {actions.shape}")
        print(f"  - pred_xyz shape: {result['pred_xyz'].shape}")
        print(f"  - pred_rot shape: {result['pred_rot'].shape}")
        print(f"  - prev_logprobs shape: {result['prev_logprobs'].shape}")
        print(f"  - prev_values shape: {result['prev_values'].shape}")
        
        # Test reward computation
        print("\nTesting reward computation...")
        pred_xyz = result["pred_xyz"]
        gt_xyz = result["gt_xyz"][:, 0, :, :]  # Remove trajectory group dim
        
        displacement = torch.norm(pred_xyz - gt_xyz, dim=-1)
        ade = displacement.mean(dim=-1)
        rewards = -ade
        
        print(f"✓ Reward computation successful")
        print(f"  - Mean ADE: {ade.mean().item():.4f}")
        print(f"  - Mean reward: {rewards.mean().item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ Inference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("Alpamayo-R1 Rollout Integration Test")
    print("=" * 80 + "\n")
    
    # Test 1: Dataset
    dataset_ok = test_dataset()
    
    # Test 2: Model loading
    model = test_model_loading()
    model_ok = model is not None
    
    # Test 3: Model inference
    if model_ok:
        inference_ok = test_model_inference(model)
    else:
        inference_ok = False
        print("\n✗ Skipping inference test (model not loaded)")
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    # print(f"Dataset:   {'✓ PASS' if dataset_ok else '✗ FAIL'}")
    print(f"Model:     {'✓ PASS' if model_ok else '✗ FAIL'}")
    print(f"Inference: {'✓ PASS' if inference_ok else '✗ FAIL'}")
    print("=" * 80)
    
    # if dataset_ok and model_ok and inference_ok:
    #     print("\n✓ All tests passed! Ready for training.")
    #     return 0
    # else:
    #     print("\n✗ Some tests failed. Please check the errors above.")
    #     return 1


if __name__ == "__main__":
    exit(main())
