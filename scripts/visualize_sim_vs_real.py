"""Visual comparison: simulated vs real microscopy images.

Generates side-by-side images showing simulated microscope views
next to real microscopy frames for visual quality assessment.

Usage:
  python scripts/visualize_sim_vs_real.py
  python scripts/visualize_sim_vs_real.py --real-dir E:\SynologyDrive\Han-Workspace\LaserSpermShoot-20250910\video2frames\frames\capture_20250910-152537 --num-compare 10
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import cv2
import numpy as np

from src.simulator.background import BackgroundGeneratorFactory
from src.simulator.environment import MicroscopeEnvironment
from src.simulator.optics import OpticsProcessor
from src.simulator.targets import get_target_generator_for_task
from src.utils.config import load_simulator_config


def load_real_images(real_dir: Path, num_images: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Load random real microscopy images, cropped to 224x224."""
    image_files = sorted(
        p for p in real_dir.glob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    )

    if not image_files:
        print(f"No images found in {real_dir}")
        return []

    indices = rng.choice(len(image_files), size=min(num_images, len(image_files)), replace=False)
    images = []

    for idx in indices:
        img = cv2.imread(str(image_files[idx]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Crop to 224x224 from center
        h, w = img.shape
        cy, cx = h // 2, w // 2
        crop = img[cy - 112:cy + 112, cx - 112:cx + 112]

        # Convert to BGR for display
        crop_bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
        images.append(crop_bgr)

    return images


def generate_simulated_images(
    task_name: str,
    num_images: int,
    sim_config_path: Path,
    background_type: str,
    background_dir: str | None,
    seed: int,
) -> list[np.ndarray]:
    """Generate simulated microscope images for comparison."""
    config = load_simulator_config(sim_config_path)
    target_gen = get_target_generator_for_task(task_name)

    bg_image_dir = Path(background_dir) if background_dir else None
    background = BackgroundGeneratorFactory.create(
        background_type=background_type, image_dir=bg_image_dir
    )

    env = MicroscopeEnvironment(
        config=config,
        target_generator=target_gen,
        background=background,
    )

    rng = np.random.default_rng(seed)
    images = []

    for i in range(num_images):
        obs, info = env.reset(seed=seed + i)
        images.append(obs)

    return images


def create_comparison_grid(
    sim_images: list[np.ndarray],
    real_images: list[np.ndarray],
    task_name: str,
) -> np.ndarray:
    """Create a side-by-side comparison grid.

    Layout:
    - Top row: simulated images
    - Bottom row: real images
    - Labels on left side
    """
    n = min(len(sim_images), len(real_images))
    if n == 0:
        return np.zeros((100, 400, 3), dtype=np.uint8)

    img_h, img_w = sim_images[0].shape[:2]
    label_width = 120
    padding = 10

    # Grid dimensions
    grid_w = label_width + n * (img_w + padding) + padding
    grid_h = 2 * (img_h + padding) + 60  # extra for title

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240  # Light gray background

    # Title
    title = f"Sim vs Real: {task_name}"
    cv2.putText(grid, title, (label_width, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Row labels
    y_sim = 50 + padding
    y_real = y_sim + img_h + padding

    cv2.putText(grid, "Simulated", (10, y_sim + img_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 0), 1)
    cv2.putText(grid, "Real", (10, y_real + img_h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)

    # Place images
    for i in range(n):
        x = label_width + i * (img_w + padding) + padding

        # Simulated
        grid[y_sim:y_sim + img_h, x:x + img_w] = sim_images[i]

        # Real
        if i < len(real_images):
            grid[y_real:y_real + img_h, x:x + img_w] = real_images[i]

    return grid


def main():
    parser = argparse.ArgumentParser(description="Visual comparison of simulated vs real microscopy images.")
    parser.add_argument("--real-dir", type=str,
                        default=r"E:\SynologyDrive\Han-Workspace\LaserSpermShoot-20250910\video2frames\frames\capture_20250910-152537",
                        help="Directory containing real microscopy frames")
    parser.add_argument("--task", type=str, default="sperm_head",
                        choices=["microsphere", "yeast", "sperm_head", "sperm_tail"],
                        help="Task to simulate")
    parser.add_argument("--num-compare", type=int, default=8,
                        help="Number of images to compare")
    parser.add_argument("--sim-config", type=str, default="configs/simulator/default.yaml",
                        help="Simulator config path")
    parser.add_argument("--background-type", type=str, default="perlin",
                        choices=["perlin", "image"],
                        help="Background type for simulation")
    parser.add_argument("--background-dir", type=str, default="data/backgrounds",
                        help="Background image directory")
    parser.add_argument("--output", type=str, default="data/comparison.png",
                        help="Output image path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print("Loading real images...")
    real_images = load_real_images(Path(args.real_dir), args.num_compare, rng)
    print(f"  Loaded {len(real_images)} real images")

    print(f"Generating simulated {args.task} images...")
    sim_config_path = Path(args.sim_config)
    if not sim_config_path.exists():
        # Try relative to project root
        sim_config_path = Path(__file__).resolve().parent.parent / args.sim_config

    sim_images = generate_simulated_images(
        task_name=args.task,
        num_images=args.num_compare,
        sim_config_path=sim_config_path,
        background_type=args.background_type,
        background_dir=args.background_dir,
        seed=args.seed,
    )
    print(f"  Generated {len(sim_images)} simulated images")

    print("Creating comparison grid...")
    grid = create_comparison_grid(sim_images, real_images, args.task)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), grid)
    print(f"Saved comparison to: {output_path}")

    # Also show statistics
    print("\n--- Image Statistics ---")
    if sim_images:
        sim_mean = np.mean([img.mean() for img in sim_images])
        sim_std = np.mean([img.std() for img in sim_images])
        print(f"Simulated: mean={sim_mean:.1f}, std={sim_std:.1f}")
    if real_images:
        real_mean = np.mean([img.mean() for img in real_images])
        real_std = np.mean([img.std() for img in real_images])
        print(f"Real:      mean={real_mean:.1f}, std={real_std:.1f}")


if __name__ == "__main__":
    main()
