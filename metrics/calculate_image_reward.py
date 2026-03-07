#!/usr/bin/env python3
"""
Calculate Image Reward (ImageReward-v1.0) score for generated images.

Usage:
    python metrics/calculate_image_reward.py \
        --gen-dir /path/to/generated/images \
        --output /path/to/results.txt \
        --metadata datasets/mjhq_5000_samples.json
"""

import argparse
import json
import os
from pathlib import Path
import torch
from tqdm import tqdm


def compute_image_reward(
    metadata_path,
    gen_dir,
    device="cuda",
    batch_size=1,
    start_idx=0,
    max_samples=None
):
    """
    Compute Image Reward score for generated images.

    Args:
        metadata_path: Path to JSON file with prompts
        gen_dir: Directory containing generated images
        device: Device to use (cuda or cpu)
        batch_size: Batch size (ImageReward processes one at a time)
        start_idx: Starting index for processing
        max_samples: Maximum number of samples to process (None for all)

    Returns:
        Dictionary with image_reward score and per-category scores
    """
    # Import ImageReward library
    try:
        import ImageReward as RM
    except ImportError:
        print("ERROR: ImageReward library not found.")
        print("Please install it with: pip install image-reward")
        raise

    print(f"\n{'='*80}")
    print(f"Computing Image Reward Score")
    print(f"{'='*80}")
    print(f"Model: ImageReward-v1.0")
    print(f"Device: {device}")
    print(f"Generated images: {gen_dir}")
    print(f"Metadata: {metadata_path}")

    # Load metadata
    print(f"\nLoading metadata...")
    with open(metadata_path, 'r') as f:
        samples = json.load(f)

    sample_list = list(samples.items())
    total_samples = len(sample_list)

    # Determine range
    end_idx = total_samples if max_samples is None else min(start_idx + max_samples, total_samples)
    print(f"Processing samples {start_idx} to {end_idx-1} ({end_idx - start_idx} images)")

    # Load ImageReward model
    print(f"\nLoading ImageReward-v1.0 model...")
    model = RM.load("ImageReward-v1.0", device=device)
    print(f"Model loaded successfully")

    # Process images
    print(f"\n{'='*80}")
    print("Computing scores...")
    print(f"{'='*80}\n")

    scores = []
    category_scores = {}
    missing_images = []
    error_images = []

    for idx in tqdm(range(start_idx, end_idx), desc="Image Reward", dynamic_ncols=True):
        img_id, info = sample_list[idx]
        category = info['category']
        prompt = info['prompt']

        # Initialize category scores
        if category not in category_scores:
            category_scores[category] = []

        # Check for both .jpg and .png extensions
        img_path_jpg = Path(gen_dir) / category / f"{img_id}.jpg"
        img_path_png = Path(gen_dir) / category / f"{img_id}.png"

        if img_path_jpg.exists():
            img_path = str(img_path_jpg)
        elif img_path_png.exists():
            img_path = str(img_path_png)
        else:
            missing_images.append(img_id)
            continue

        try:
            # Compute score
            with torch.inference_mode():
                score = model.score(prompt, img_path)

            scores.append(score)
            category_scores[category].append(score)

        except Exception as e:
            error_images.append((img_id, str(e)))
            print(f"\nError processing {img_id}: {e}")
            continue

    # Compute statistics
    if len(scores) == 0:
        print("\nERROR: No images were successfully processed!")
        return None

    avg_score = sum(scores) / len(scores)

    # Compute per-category averages
    category_averages = {}
    for category, cat_scores in category_scores.items():
        if len(cat_scores) > 0:
            category_averages[category] = sum(cat_scores) / len(cat_scores)

    # Print results
    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"Image Reward Score: {avg_score:.4f}")
    print(f"\nProcessed: {len(scores)}/{end_idx - start_idx} images")

    if missing_images:
        print(f"Missing images: {len(missing_images)}")
    if error_images:
        print(f"Errors: {len(error_images)}")

    print(f"\nPer-category scores:")
    for category in sorted(category_averages.keys()):
        count = len(category_scores[category])
        avg = category_averages[category]
        print(f"  {category:12s}: {avg:7.4f} ({count} images)")

    print(f"{'='*80}\n")

    # Return results
    results = {
        "image_reward": avg_score,
        "total_processed": len(scores),
        "total_requested": end_idx - start_idx,
        "missing_images": len(missing_images),
        "errors": len(error_images),
        "category_scores": category_averages
    }

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate Image Reward score for generated images (SVDQuant configuration)"
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default='datasets/mjhq_5000_samples.json',
        help='Path to metadata JSON file with prompts'
    )
    parser.add_argument(
        '--gen-dir',
        type=str,
        required=True,
        help='Directory containing generated images'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--start-idx',
        type=int,
        default=0,
        help='Starting index for processing (default: 0)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to process (default: all)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output file for results'
    )

    args = parser.parse_args()

    # Compute Image Reward
    results = compute_image_reward(
        metadata_path=args.metadata,
        gen_dir=args.gen_dir,
        device=args.device,
        start_idx=args.start_idx,
        max_samples=args.max_samples
    )

    if results is None:
        print("Failed to compute Image Reward score")
        return 1

    # Save results
    with open(args.output, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Image Reward Results\n")
        f.write(f"{'='*80}\n")
        f.write(f"Metadata: {args.metadata}")
        f.write(f"Generated images: {args.gen_dir}\n")
        f.write(f"Image Reward Score: {results['image_reward']:.4f}\n")
        f.write(f"Processed: {results['total_processed']}/{results['total_requested']} images\n")
        f.write(f"\nPer-category scores:\n")
        for category in sorted(results['category_scores'].keys()):
            score = results['category_scores'][category]
            f.write(f"  {category:12s}: {score:.4f}\n")
        f.write(f"{'='*80}\n")

    print(f"Results saved to {args.output}")

    return 0


if __name__ == '__main__':
    exit(main())
