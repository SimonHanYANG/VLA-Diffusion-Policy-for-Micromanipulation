"""Extract clean background patches from real microscopy video frames.

Reads sperm video frames, detects and excludes regions with targets,
and saves clean 224x224 background patches to data/backgrounds/.

Usage:
  python scripts/extract_backgrounds.py --source-dir E:\SynologyDrive\Han-Workspace\LaserSpermShoot-20250910\video2frames\frames\capture_20250910-152537
  python scripts/extract_backgrounds.py --source-dir /path/to/frames --patch-size 224 --num-frames 200 --patches-per-frame 5
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import cv2
import numpy as np
from tqdm import tqdm


def detect_target_mask(gray: np.ndarray, threshold: int = 120) -> np.ndarray:
    """Create a binary mask of target regions (dark objects).

    Real microscopy images have uniform gray background (~146) with
    dark targets (~50-80). We threshold to find target regions.

    Returns:
        (H, W) bool array: True = target region (exclude), False = clean background.
    """
    # Targets are darker than background
    target_mask = gray < threshold

    # Also detect very bright spots (debris, reflections)
    bright_mask = gray > 220

    # Dilate to exclude regions near targets (edge effects)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    target_mask = cv2.dilate(target_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    bright_mask = cv2.dilate(bright_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    return target_mask | bright_mask


def find_clean_patches(
    gray: np.ndarray,
    patch_size: int,
    num_patches: int,
    exclude_mask: np.ndarray,
    rng: np.random.Generator,
    max_attempts: int = 100,
) -> list[np.ndarray]:
    """Find random clean patches that don't overlap with target regions.

    Args:
        gray: (H, W) grayscale image.
        patch_size: Size of square patch to extract.
        num_patches: Number of patches to find.
        exclude_mask: (H, W) bool, True = exclude this region.
        rng: Random generator.
        max_attempts: Max random tries per patch.

    Returns:
        List of (patch_size, patch_size) uint8 arrays.
    """
    h, w = gray.shape
    patches = []

    # Compute a "clean score" for each possible top-left corner
    # A patch is clean if none of its pixels are in exclude_mask
    # Use a summed area table for fast checking
    exclude_float = exclude_mask.astype(np.float32)
    sat = cv2.integral(exclude_float)  # (H+1, W+1)

    def patch_excluded_count(y, x):
        """Count excluded pixels in patch starting at (y, x)."""
        y2, x2 = y + patch_size, x + patch_size
        return (
            sat[y2, x2] - sat[y, x2] - sat[y2, x] + sat[y, x]
        )

    # Find all valid top-left corners (no excluded pixels)
    valid_positions = []
    for y in range(0, h - patch_size + 1, patch_size // 4):
        for x in range(0, w - patch_size + 1, patch_size // 4):
            if patch_excluded_count(y, x) == 0:
                valid_positions.append((y, x))

    if not valid_positions:
        return patches

    # Randomly sample from valid positions
    indices = rng.choice(len(valid_positions), size=min(num_patches, len(valid_positions)), replace=False)
    for idx in indices:
        y, x = valid_positions[idx]
        patch = gray[y:y + patch_size, x:x + patch_size].copy()
        patches.append(patch)

    return patches


def main():
    parser = argparse.ArgumentParser(description="Extract background patches from real microscopy frames.")
    parser.add_argument("--source-dir", type=str, required=True,
                        help="Directory containing microscopy frames (jpg)")
    parser.add_argument("--output-dir", type=str, default="data/backgrounds",
                        help="Output directory for background patches")
    parser.add_argument("--patch-size", type=int, default=640,
                        help="Size of square patches to extract (should match simulator image_size)")
    parser.add_argument("--num-frames", type=int, default=200,
                        help="Number of frames to sample from")
    parser.add_argument("--patches-per-frame", type=int, default=5,
                        help="Number of patches to extract per frame")
    parser.add_argument("--threshold", type=int, default=120,
                        help="Pixel threshold for target detection (lower = stricter)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all image files
    image_files = sorted(
        p for p in source_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    if not image_files:
        print(f"No images found in {source_dir}")
        return

    print(f"Found {len(image_files)} images in {source_dir}")

    # Sample frames evenly
    rng = np.random.default_rng(args.seed)
    num_frames = min(args.num_frames, len(image_files))
    indices = np.linspace(0, len(image_files) - 1, num_frames, dtype=int)
    selected_files = [image_files[i] for i in indices]

    print(f"Sampling {num_frames} frames, extracting {args.patches_per_frame} patches each")
    print(f"Patch size: {args.patch_size}x{args.patch_size}")
    print(f"Target detection threshold: < {args.threshold}")

    total_patches = 0
    for file_path in tqdm(selected_files, desc="Processing frames"):
        # Read as grayscale
        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Detect target regions
        exclude_mask = detect_target_mask(img, threshold=args.threshold)

        # Find clean patches
        patches = find_clean_patches(
            img, args.patch_size, args.patches_per_frame, exclude_mask, rng
        )

        # Save patches
        for patch in patches:
            patch_name = f"bg_{total_patches:05d}.png"
            cv2.imwrite(str(output_dir / patch_name), patch)
            total_patches += 1

    print(f"\nDone! Extracted {total_patches} background patches to {output_dir}")
    print(f"Clean rate: {total_patches / (num_frames * args.patches_per_frame) * 100:.1f}%")


if __name__ == "__main__":
    main()
