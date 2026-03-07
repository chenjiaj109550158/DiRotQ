#!/usr/bin/env python3
"""
Calculate PSNR, LPIPS, SSIM between reference and generated images.

Usage:
    python metrics/calculate_psnr_lpips_ssim.py \
        --ref-dir /path/to/reference/images \
        --gen-dir /path/to/generated/images \
        --output /path/to/results.txt
"""

import argparse
import os
import tempfile
from pathlib import Path

import torch
import torchmetrics
import torchvision
from PIL import Image
from torch.utils import data
# Adapted from deepcompressor (SVDQuant) similarity metrics
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm


def create_flat_symlinks(src_dir, dst_dir):
    """Flatten nested image dirs into a single dir of symlinks."""
    src = Path(src_dir)
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)

    images = sorted(src.rglob("*.png")) + sorted(src.rglob("*.jpg"))
    for img in tqdm(images, desc=f"Symlinking {src.name}"):
        rel = img.relative_to(src)
        flat_name = str(rel).replace(os.sep, "_")
        link = dst / flat_name
        if not link.exists():
            os.symlink(img.absolute(), link)
    return len(images)


class PairedImageDataset(data.Dataset):
    """Dataset that loads paired (gen, ref) images from two flat directories."""

    def __init__(self, gen_dirpath: str, ref_dirpath: str):
        super().__init__()
        self.gen_names = sorted(
            name for name in os.listdir(gen_dirpath) if name.endswith(".png") or name.endswith(".jpg")
        )
        self.ref_names = sorted(
            name for name in os.listdir(ref_dirpath) if name.endswith(".png") or name.endswith(".jpg")
        )
        assert len(self.ref_names) == len(self.gen_names), (
            f"Ref ({len(self.ref_names)}) and gen ({len(self.gen_names)}) image counts differ"
        )
        self.gen_dirpath = gen_dirpath
        self.ref_dirpath = ref_dirpath
        self.transform = torchvision.transforms.ToTensor()

    def __len__(self):
        return len(self.ref_names)

    def __getitem__(self, idx: int):
        name = self.ref_names[idx]
        assert name == self.gen_names[idx]
        ref_image = Image.open(os.path.join(self.ref_dirpath, name)).convert("RGB")
        gen_image = Image.open(os.path.join(self.gen_dirpath, name)).convert("RGB")
        if ref_image.size != gen_image.size:
            ref_image = ref_image.resize(gen_image.size, Image.Resampling.BICUBIC)
        return [self.transform(gen_image), self.transform(ref_image)]


def compute_image_similarity_metrics(
    ref_dirpath: str,
    gen_dirpath: str,
    metric_names: tuple[str, ...] = ("psnr", "lpips", "ssim"),
    batch_size: int = 64,
    num_workers: int = 0,
    device: str = "cuda",
) -> dict[str, float]:
    """Compute PSNR/LPIPS/SSIM between paired images. Adapted from deepcompressor (SVDQuant)."""
    metrics: dict[str, torchmetrics.Metric] = {}
    for name in metric_names:
        if name == "psnr":
            m = PeakSignalNoiseRatio(data_range=(0, 1), reduction="elementwise_mean", dim=(1, 2, 3))
        elif name == "lpips":
            m = LearnedPerceptualImagePatchSimilarity(normalize=True)
        elif name == "ssim":
            m = StructuralSimilarityIndexMeasure(data_range=(0, 1))
        else:
            raise ValueError(f"Unknown metric: {name}")
        metrics[name] = m.to(device)

    dataset = PairedImageDataset(gen_dirpath, ref_dirpath)
    dataloader = data.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing similarity metrics"):
            batch = [t.to(device) for t in batch]
            for m in metrics.values():
                m.update(batch[0], batch[1])

    return {name: m.compute().item() for name, m in metrics.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Calculate PSNR, LPIPS, SSIM between reference and generated images"
    )
    parser.add_argument('--ref-dir', type=str, required=True,
                        help='Directory containing reference images')
    parser.add_argument('--gen-dir', type=str, required=True,
                        help='Directory containing generated images')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to write results')

    args = parser.parse_args()

    print("=" * 70)
    print("PSNR / LPIPS / SSIM")
    print("=" * 70)
    print(f"Reference : {args.ref_dir}")
    print(f"Generated : {args.gen_dir}")

    with tempfile.TemporaryDirectory() as tmp_ref, tempfile.TemporaryDirectory() as tmp_gen:
        n_ref = create_flat_symlinks(args.ref_dir, tmp_ref)
        n_gen = create_flat_symlinks(args.gen_dir, tmp_gen)

        print(f"\nReference images : {n_ref}")
        print(f"Generated images : {n_gen}")
        if n_ref != n_gen:
            print("WARNING: counts differ!")

        print("\nComputing metrics...")
        results = compute_image_similarity_metrics(
            ref_dirpath=tmp_ref,
            gen_dirpath=tmp_gen,
        )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k.upper():6s}: {v:.4f}")
    print("=" * 70)

    results_path = Path(args.output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "a") as f:
        for k, v in results.items():
            f.write(f"{k.upper():6s}: {v:.4f}\n")
        f.write(f"  Reference: {args.ref_dir}\n")
        f.write(f"  Generated: {args.gen_dir}\n")
        f.write("=" * 70 + "\n")

    print(f"\nResults appended to {results_path}")


if __name__ == "__main__":
    main()
