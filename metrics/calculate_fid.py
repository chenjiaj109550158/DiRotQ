#!/usr/bin/env python3
"""
Calculate FID score between reference and generated images.

Usage:
    python metrics/calculate_fid.py \
        --ref-dir /path/to/reference/images \
        --gen-dir /path/to/generated/images \
        --output /path/to/results.txt
"""

import argparse
import os
import tempfile
from pathlib import Path
from cleanfid import fid
import numpy as np
from tqdm import tqdm


def create_flat_symlinks(src_dir, temp_dir, desc="Creating symlinks"):
    """Flatten nested image dirs into a single dir of symlinks (cleanfid needs flat dirs)."""
    src_path = Path(src_dir)
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(src_path.rglob(ext))

    print(f"Found {len(image_files)} images in {src_dir}")

    # Create symlinks in flat structure
    for img_path in tqdm(image_files, desc=desc):
        # Use relative path from src_dir to create unique names
        rel_path = img_path.relative_to(src_path)
        # Replace directory separators with underscores to flatten
        flat_name = str(rel_path).replace(os.sep, '_')
        symlink_path = temp_path / flat_name

        # Create symlink
        if not symlink_path.exists():
            os.symlink(img_path.absolute(), symlink_path)

    return temp_path


def compute_fid_score(
    ref_dir,
    gen_dir,
    mode="clean",
    batch_size=64,
    device="cuda",
    ref_cache=None,
    gen_cache=None,
    verbose=True
):
    """
    Compute FID score between reference and generated images.

    Args:
        ref_dir: Directory containing reference images
        gen_dir: Directory containing generated images
        mode: FID computation mode ("clean" for InceptionV3, "legacy_pytorch", etc.)
        batch_size: Batch size for feature extraction
        device: Device to use (cuda or cpu)
        ref_cache: Path to cache reference features (optional)
        gen_cache: Path to cache generated features (optional)
        verbose: Whether to print verbose output

    Returns:
        FID score (float)
    """
    print(f"\n{'='*80}")
    print(f"Computing FID Score")
    print(f"{'='*80}")
    print(f"Mode: {mode}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Reference dir: {ref_dir}")
    print(f"Generated dir: {gen_dir}")

    # Create temporary flat directories for both reference and generated images
    print(f"\n{'='*80}")
    print("Preparing image directories...")
    print(f"{'='*80}")

    with tempfile.TemporaryDirectory() as temp_ref, tempfile.TemporaryDirectory() as temp_gen:
        # Create flat symlink directories
        flat_ref_dir = create_flat_symlinks(ref_dir, temp_ref, "Preparing reference images")
        flat_gen_dir = create_flat_symlinks(gen_dir, temp_gen, "Preparing generated images")

        # Count images
        ref_count = len(list(flat_ref_dir.glob('*')))
        gen_count = len(list(flat_gen_dir.glob('*')))

        print(f"\nReference images: {ref_count}")
        print(f"Generated images: {gen_count}")

        if ref_count != gen_count:
            print(f"\n⚠️  WARNING: Number of reference ({ref_count}) and generated ({gen_count}) images don't match!")

        # Extract features and compute FID
        print(f"\n{'='*80}")
        print("Computing FID...")
        print(f"{'='*80}")

        # Load or compute reference features
        if ref_cache and os.path.exists(ref_cache):
            print(f"Loading cached reference features from {ref_cache}")
            ref_stats = np.load(ref_cache)
            mu_ref = ref_stats['mu']
            sigma_ref = ref_stats['sigma']
        else:
            print("Extracting features from reference images...")
            feat_model = fid.build_feature_extractor(mode, device)
            np_feats_ref = fid.get_folder_features(
                str(flat_ref_dir),
                feat_model,
                num_workers=0,
                batch_size=batch_size,
                device=device,
                mode=mode,
                verbose=verbose,
                description="Reference features"
            )
            mu_ref = np.mean(np_feats_ref, axis=0)
            sigma_ref = np.cov(np_feats_ref, rowvar=False)

            if ref_cache:
                print(f"Saving reference features to {ref_cache}")
                os.makedirs(os.path.dirname(ref_cache), exist_ok=True)
                np.savez(ref_cache, mu=mu_ref, sigma=sigma_ref)

        # Load or compute generated features
        if gen_cache and os.path.exists(gen_cache):
            print(f"Loading cached generated features from {gen_cache}")
            gen_stats = np.load(gen_cache)
            mu_gen = gen_stats['mu']
            sigma_gen = gen_stats['sigma']
        else:
            print("Extracting features from generated images...")
            feat_model = fid.build_feature_extractor(mode, device)
            np_feats_gen = fid.get_folder_features(
                str(flat_gen_dir),
                feat_model,
                num_workers=0,
                batch_size=batch_size,
                device=device,
                mode=mode,
                verbose=verbose,
                description="Generated features"
            )
            mu_gen = np.mean(np_feats_gen, axis=0)
            sigma_gen = np.cov(np_feats_gen, rowvar=False)

            if gen_cache:
                print(f"Saving generated features to {gen_cache}")
                os.makedirs(os.path.dirname(gen_cache), exist_ok=True)
                np.savez(gen_cache, mu=mu_gen, sigma=sigma_gen)

        # Compute FID
        fid_score = fid.frechet_distance(mu_ref, sigma_ref, mu_gen, sigma_gen)
        fid_score = float(fid_score)

        return fid_score


def main():
    parser = argparse.ArgumentParser(
        description="Calculate FID score between reference and generated images (SVDQuant configuration)"
    )
    parser.add_argument(
        '--ref-dir',
        type=str,
        required=True,
        help='Directory containing reference images (can have subdirectories)'
    )
    parser.add_argument(
        '--gen-dir',
        type=str,
        required=True,
        help='Directory containing generated images (can have subdirectories)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to write FID results'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='clean',
        choices=['clean', 'legacy_pytorch', 'legacy_tensorflow'],
        help='FID computation mode (default: clean, same as SVDQuant)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for feature extraction (default: 64)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda)'
    )
    parser.add_argument(
        '--ref-cache',
        type=str,
        default=None,
        help='Path to cache reference features (optional)'
    )
    parser.add_argument(
        '--gen-cache',
        type=str,
        default=None,
        help='Path to cache generated features (optional)'
    )
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable feature caching'
    )

    args = parser.parse_args()

    # Disable caching if requested
    ref_cache = None if args.no_cache else args.ref_cache
    gen_cache = None if args.no_cache else args.gen_cache

    # Compute FID
    fid_score = compute_fid_score(
        ref_dir=args.ref_dir,
        gen_dir=args.gen_dir,
        mode=args.mode,
        batch_size=args.batch_size,
        device=args.device,
        ref_cache=ref_cache,
        gen_cache=gen_cache,
        verbose=True
    )

    print(f"\n{'='*80}")
    print(f"RESULTS")
    print(f"{'='*80}")
    print(f"FID Score: {fid_score:.4f}")
    print(f"{'='*80}\n")

    # Save results
    results_path = Path(args.output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'a') as f:
        f.write(f"FID Score (mode={args.mode}): {fid_score:.4f}\n")
        f.write(f"  Reference: {args.ref_dir}\n")
        f.write(f"  Generated: {args.gen_dir}\n")
        f.write(f"  {'='*60}\n")

    print(f"Results appended to {results_path}")


if __name__ == '__main__':
    main()
