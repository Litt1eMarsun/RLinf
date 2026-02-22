#!/usr/bin/env python3
"""
Preload PhysicalAI-AV dataset clips to local cache.

This script downloads dataset clips to avoid streaming during training.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def preload_clip(clip_id: str, avdi, skip_cameras: bool = False):
    """
    Preload all data needed for a single clip using physical_ai_av's download_clip_features.
    
    This is the CORRECT way to download data from HuggingFace Hub to local cache.
    
    Args:
        clip_id: The clip ID to preload
        avdi: PhysicalAIAVDatasetInterface instance
        skip_cameras: If True, skip camera data preloading
    """
    print(f"Preloading clip: {clip_id}")
    
    # Build list of features to download
    features_to_download = ['egomotion']
    
    if not skip_cameras:
        cameras = [
            "camera_front_wide_120fov",
            "camera_front_tele_30fov",
            "camera_cross_left_120fov",
            "camera_cross_right_120fov",
        ]
        features_to_download.extend(cameras)
    
    # Use physical_ai_av's built-in download method
    # This is the CORRECT way - it handles file paths and caching properly
    try:
        avdi.download_clip_features(clip_id=clip_id, features=features_to_download)
        print(f"✓ Clip {clip_id} downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download clip {clip_id}: {e}")
        return False


def load_downloaded_clips(cache_dir: str) -> set[str]:
    """Load the set of already downloaded clip IDs from tracking file."""
    tracking_file = os.path.join(cache_dir, ".downloaded_clips.txt")
    if not os.path.exists(tracking_file):
        return set()
    
    try:
        with open(tracking_file, 'r') as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"Warning: Failed to read tracking file: {e}")
        return set()


def mark_clip_downloaded(cache_dir: str, clip_id: str):
    """Mark a clip as successfully downloaded."""
    tracking_file = os.path.join(cache_dir, ".downloaded_clips.txt")
    try:
        with open(tracking_file, 'a') as f:
            f.write(f"{clip_id}\n")
    except Exception as e:
        print(f"Warning: Failed to update tracking file: {e}")


def load_clip_ids_from_parquet(parquet_path: str, max_clips: int = 1280) -> list[str]:
    """Load clip IDs from parquet file."""
    print(f"Loading clip IDs from: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    
    # Check if 'clip_id' column exists
    if 'clip_id' in df.columns:
        clip_ids = df['clip_id'].tolist()[:max_clips]
    else:
        # Try to get from index
        df_reset = df.reset_index()
        if 'clip_id' in df_reset.columns:
            # Filter for train split
            train_df = df_reset[df_reset['split'] == 'train']
            clip_ids = train_df['clip_id'].tolist()[:max_clips]
        else:
            raise ValueError(f"Could not find 'clip_id' column in {parquet_path}")
    
    print(f"✓ Loaded {len(clip_ids)} clip IDs")
    return clip_ids


def main():
    parser = argparse.ArgumentParser(
        description="Preload PhysicalAI-AV dataset clips to local cache"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data/workspace/RLinf/data/clip_index.parquet",
        help="Path to clip index parquet file",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/data/workspace/RLinf/data",
        help="Cache directory for HuggingFace Hub",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=128,
        help="Maximum number of clips to preload",
    )
    parser.add_argument(
        "--skip-cameras",
        action="store_true",
        help="Skip camera data (only download egomotion)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous download (skip already downloaded clips)",
    )
    
    args = parser.parse_args()
    
    # Set HuggingFace cache environment variables
    os.environ['HF_HOME'] = args.cache_dir
    os.environ['HF_HUB_CACHE'] = os.path.join(args.cache_dir, 'hub')
    
    print(f"\n{'='*80}")
    print(f"PhysicalAI-AV Dataset Preloader")
    print(f"{'='*80}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"HF Hub cache: {os.environ['HF_HUB_CACHE']}")
    print(f"Max clips: {args.max_clips}")
    print(f"Skip cameras: {args.skip_cameras}")
    print(f"Resume: {args.resume}")
    print(f"{'='*80}\n")
    
    # Import physical_ai_av after setting environment variables
    try:
        import physical_ai_av
    except ImportError:
        print("Error: physical_ai_av is not installed!")
        print("Install it with: pip install physical_ai_av")
        sys.exit(1)
    
    # Load clip IDs
    try:
        clip_ids = load_clip_ids_from_parquet(args.data_path, args.max_clips)
    except Exception as e:
        print(f"Error loading clip IDs: {e}")
        sys.exit(1)
    
    # Initialize PhysicalAIAVDatasetInterface
    print("\nInitializing PhysicalAI-AV dataset interface...")
    try:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface(
            cache_dir=args.cache_dir
        )
        print(f"✓ Connected to: {avdi.repo_id}")
        print(f"  Revision: {avdi.revision}")
    except Exception as e:
        print(f"Error initializing dataset interface: {e}")
        sys.exit(1)
    
    # Load already downloaded clips if resuming
    downloaded_clips = set()
    if args.resume:
        downloaded_clips = load_downloaded_clips(args.cache_dir)
        print(f"\n✓ Resuming: {len(downloaded_clips)} clips already downloaded")
        clip_ids = [cid for cid in clip_ids if cid not in downloaded_clips]
        print(f"  Remaining: {len(clip_ids)} clips to download")
    
    if not clip_ids:
        print("\nNo clips to download!")
        return
    
    # Download clips
    print(f"\n{'='*80}")
    print(f"Starting download of {len(clip_ids)} clips")
    print(f"{'='*80}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, clip_id in enumerate(clip_ids, 1):
        print(f"\n[{i}/{len(clip_ids)}] ", end="")
        
        try:
            if preload_clip(clip_id, avdi, skip_cameras=args.skip_cameras):
                mark_clip_downloaded(args.cache_dir, clip_id)
                success_count += 1
            else:
                fail_count += 1
        except KeyboardInterrupt:
            print("\n\nInterrupted by user!")
            break
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            fail_count += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"Download Summary")
    print(f"{'='*80}")
    print(f"Success: {success_count}/{len(clip_ids)}")
    print(f"Failed: {fail_count}/{len(clip_ids)}")
    print(f"{'='*80}\n")
    
    if fail_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
