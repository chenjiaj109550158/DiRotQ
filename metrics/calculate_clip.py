#!/usr/bin/env python3
"""
Calculate CLIP Score and CLIP IQA for generated images.

Usage:
    python metrics/calculate_clip.py \
        --gen-dir /path/to/generated/images \
        --output /path/to/results.txt \
        --metadata datasets/mjhq_5000_samples.json
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchmetrics.multimodal import CLIPImageQualityAssessment, CLIPScore
from tqdm import tqdm


def compute_clip_metrics(
    metadata_path: str,
    gen_dir: str,
    device: str = "cuda",
    batch_size: int = 16,
) -> dict[str, float]:
    """Compute CLIP Score (text-image alignment) and CLIP IQA (perceptual quality)."""

    print(f"\n{'='*70}")
    print("Computing CLIP Score & CLIP IQA")
    print(f"{'='*70}")
    print(f"Generated images: {gen_dir}")
    print(f"Metadata: {metadata_path}")

    # Load metadata
    with open(metadata_path, "r") as f:
        samples = json.load(f)
    sample_list = list(samples.items())

    # Collect valid (image_tensor, prompt) pairs
    # Use uint8 [0,255] tensors (same as SVDQuant's multimodal.py)
    image_tensors = []
    prompts = []

    for img_id, info in tqdm(sample_list, desc="Loading images"):
        category = info["category"]
        prompt = info["prompt"]

        img_path_jpg = Path(gen_dir) / category / f"{img_id}.jpg"
        img_path_png = Path(gen_dir) / category / f"{img_id}.png"

        if img_path_jpg.exists():
            img_path = img_path_jpg
        elif img_path_png.exists():
            img_path = img_path_png
        else:
            continue

        img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # uint8 [0,255] CHW
        image_tensors.append(img_tensor)
        prompts.append(prompt)

    print(f"Loaded {len(image_tensors)} images")

    # Initialize metrics (same models as SVDQuant)
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)
    clip_iqa = CLIPImageQualityAssessment(model_name_or_path="openai/clip-vit-large-patch14").to(device)

    # Process in batches
    n = len(image_tensors)
    with torch.no_grad():
        for i in tqdm(range(0, n, batch_size), desc="Computing CLIP metrics"):
            batch_imgs = torch.stack(image_tensors[i : i + batch_size]).to(device)
            batch_prompts = prompts[i : i + batch_size]

            clip_score.update(batch_imgs, batch_prompts)
            clip_iqa.update(batch_imgs.to(torch.float32))

    results = {
        "clip_score": clip_score.compute().item(),
        "clip_iqa": clip_iqa.compute().mean().item(),
    }

    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    for k, v in results.items():
        print(f"  {k.upper():12s}: {v:.4f}")
    print(f"{'='*70}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Calculate CLIP Score and CLIP IQA for generated images"
    )
    parser.add_argument("--gen-dir", type=str, required=True,
                        help="Directory containing generated images")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file for results")
    parser.add_argument("--metadata", type=str, default="datasets/mjhq_5000_samples.json",
                        help="Path to metadata JSON file with prompts")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"], help="Device to use (default: cuda)")

    args = parser.parse_args()

    results = compute_clip_metrics(
        metadata_path=args.metadata,
        gen_dir=args.gen_dir,
        device=args.device,
        batch_size=args.batch_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        for k, v in results.items():
            f.write(f"{k.upper():12s}: {v:.4f}\n")
        f.write(f"  Metadata: {args.metadata}")
        f.write(f"  Generated: {args.gen_dir}\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults appended to {output_path}")


if __name__ == "__main__":
    main()
