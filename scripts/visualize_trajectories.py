"""Visualize generated trajectories.

Shows trajectory frames, path overlay, and statistics.

Usage:
  python scripts/visualize_trajectories.py --traj-dir data/trajectories/sperm_head/traj_00000
  python scripts/visualize_trajectories.py --task-dir data/trajectories/sperm_head --num-show 6
  python scripts/visualize_trajectories.py --task-dir data/trajectories/sperm_head --animate
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import cv2
import json
import numpy as np


def load_trajectory(traj_dir: Path) -> dict:
    """Load a single trajectory from directory."""
    meta_path = traj_dir / "meta.json"
    actions_path = traj_dir / "actions.npy"
    positions_path = traj_dir / "positions.npy"
    images_dir = traj_dir / "images"

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {traj_dir}")

    with open(meta_path) as f:
        meta = json.load(f)

    actions = np.load(actions_path) if actions_path.exists() else None
    positions = np.load(positions_path) if positions_path.exists() else None

    # Load images
    image_files = sorted(images_dir.glob("*.jpg"))
    images = [cv2.imread(str(f)) for f in image_files]

    return {
        "meta": meta,
        "actions": actions,
        "positions": positions,
        "images": images,
        "dir": traj_dir,
    }


def visualize_single_trajectory(traj: dict, output_path: Path | None = None, frame_size: int = 224):
    """Show a single trajectory with frames and path overlay."""
    images = traj["images"]
    positions = traj["positions"]
    meta = traj["meta"]
    n_frames = len(images)

    if n_frames == 0:
        print("No images in trajectory")
        return

    img_h, img_w = images[0].shape[:2]

    # Create grid: show every N-th frame
    if n_frames <= 8:
        indices = list(range(n_frames))
    else:
        indices = np.linspace(0, n_frames - 1, 8, dtype=int).tolist()

    # Top row: selected frames
    frame_w, frame_h = frame_size, frame_size
    padding = 5
    grid_w = len(indices) * (frame_w + padding) + padding
    grid_h = frame_h + 80 + 224 + 40  # frames + info + path map + margin

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    # Title
    title = f"Trajectory: {meta.get('task', 'unknown')} | Steps: {meta.get('num_steps', '?')} | Seed: {meta.get('seed', '?')}"
    cv2.putText(grid, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Place frames
    for i, idx in enumerate(indices):
        x = padding + i * (frame_w + padding)
        y = 35
        frame = cv2.resize(images[idx], (frame_w, frame_h))
        grid[y:y + frame_h, x:x + frame_w] = frame

        # Frame number
        cv2.putText(grid, f"f{idx}", (x + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Bottom: trajectory path visualization
    path_y_start = 35 + frame_h + 10
    path_h = 200
    path_w = grid_w - 2 * padding
    path_img = np.ones((path_h, path_w, 3), dtype=np.uint8) * 200

    if positions is not None and len(positions) > 0:
        # Scale positions to fit path visualization
        pos = positions.copy()
        pos_min = pos.min(axis=0)
        pos_max = pos.max(axis=0)
        pos_range = pos_max - pos_min
        if pos_range.max() > 0:
            scale = min(path_w, path_h) * 0.8 / max(pos_range.max(), 1)
            offset = np.array([path_w / 2, path_h / 2]) - (pos.mean(axis=0) - pos_min) * scale
            scaled_pos = (pos - pos_min) * scale + offset

            # Draw path
            for i in range(1, len(scaled_pos)):
                pt1 = tuple(scaled_pos[i - 1].astype(int))
                pt2 = tuple(scaled_pos[i].astype(int))
                # Color gradient: green (start) to red (end)
                ratio = i / len(scaled_pos)
                color = (0, int(255 * (1 - ratio)), int(255 * ratio))
                cv2.line(path_img, pt1, pt2, color, 2)

            # Draw start and end points
            cv2.circle(path_img, tuple(scaled_pos[0].astype(int)), 6, (0, 255, 0), -1)
            cv2.circle(path_img, tuple(scaled_pos[-1].astype(int)), 6, (0, 0, 255), -1)

            # Draw laser position (center of image)
            laser_x, laser_y = img_w // 2, img_h // 2
            laser_scaled = (np.array([laser_x, laser_y]) - pos_min) * scale + offset
            cv2.drawMarker(path_img, tuple(laser_scaled.astype(int)), (0, 0, 255),
                           cv2.MARKER_CROSS, 15, 2)

        # Info text
        info_text = f"Start: ({pos[0][0]:.0f}, {pos[0][1]:.0f}) -> End: ({pos[-1][0]:.0f}, {pos[-1][1]:.0f})"
        cv2.putText(path_img, info_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Legend
        cv2.circle(path_img, (10, path_h - 20), 5, (0, 255, 0), -1)
        cv2.putText(path_img, "Start", (20, path_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.circle(path_img, (70, path_h - 20), 5, (0, 0, 255), -1)
        cv2.putText(path_img, "End", (80, path_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        cv2.drawMarker(path_img, (130, path_h - 20), (0, 0, 255), cv2.MARKER_CROSS, 10, 2)
        cv2.putText(path_img, "Laser", (140, path_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    grid[path_y_start:path_y_start + path_h, padding:padding + path_w] = path_img
    cv2.putText(grid, "Target Trajectory (green=start, red=end)", (padding, path_y_start - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    if output_path:
        cv2.imwrite(str(output_path), grid)
        print(f"Saved to: {output_path}")
    else:
        cv2.imshow("Trajectory", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_trajectory_grid(task_dir: Path, num_show: int = 6, output_path: Path | None = None, frame_size: int = 224):
    """Show multiple trajectories in a grid."""
    traj_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("traj_")])

    if not traj_dirs:
        print(f"No trajectories found in {task_dir}")
        return

    traj_dirs = traj_dirs[:num_show]
    print(f"Showing {len(traj_dirs)} trajectories")

    # Load all trajectories
    trajectories = []
    for d in traj_dirs:
        try:
            traj = load_trajectory(d)
            trajectories.append(traj)
        except Exception as e:
            print(f"Failed to load {d}: {e}")

    if not trajectories:
        return

    # Create grid: each row is a trajectory (first frame + last frame + path info)
    frame_w, frame_h = frame_size, frame_size
    padding = 5
    cols = 3  # first frame, last frame, info
    grid_w = cols * (frame_w + padding) + 200  # extra for text
    grid_h = len(trajectories) * (frame_h + padding) + 40

    grid = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

    # Title
    task_name = trajectories[0]["meta"].get("task", "unknown")
    cv2.putText(grid, f"Trajectories: {task_name} ({len(trajectories)} shown)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    for i, traj in enumerate(trajectories):
        y = 40 + i * (frame_h + padding)
        images = traj["images"]
        meta = traj["meta"]
        positions = traj["positions"]

        # First frame
        if images:
            frame0 = cv2.resize(images[0], (frame_w, frame_h))
            grid[y:y + frame_h, padding:padding + frame_w] = frame0
            cv2.putText(grid, "First", (padding + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Last frame
        if len(images) > 1:
            frame_last = cv2.resize(images[-1], (frame_w, frame_h))
            x_last = padding + frame_w + padding
            grid[y:y + frame_h, x_last:x_last + frame_w] = frame_last
            cv2.putText(grid, "Last", (x_last + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # Path visualization (small)
        x_path = 2 * (frame_w + padding) + padding
        path_img = np.ones((frame_h, frame_w, 3), dtype=np.uint8) * 200

        if positions is not None and len(positions) > 1:
            pos = positions.copy()
            pos_min = pos.min(axis=0)
            pos_max = pos.max(axis=0)
            pos_range = pos_max - pos_min
            if pos_range.max() > 0:
                scale = frame_w * 0.7 / max(pos_range.max(), 1)
                offset = np.array([frame_w / 2, frame_h / 2]) - (pos.mean(axis=0) - pos_min) * scale
                scaled = (pos - pos_min) * scale + offset

                for j in range(1, len(scaled)):
                    pt1 = tuple(scaled[j - 1].astype(int))
                    pt2 = tuple(scaled[j].astype(int))
                    ratio = j / len(scaled)
                    color = (0, int(255 * (1 - ratio)), int(255 * ratio))
                    cv2.line(path_img, pt1, pt2, color, 1)

                cv2.circle(path_img, tuple(scaled[0].astype(int)), 4, (0, 255, 0), -1)
                cv2.circle(path_img, tuple(scaled[-1].astype(int)), 4, (0, 0, 255), -1)

        grid[y:y + frame_h, x_path:x_path + frame_w] = path_img
        cv2.putText(grid, "Path", (x_path + 2, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Info text
        x_info = x_path + frame_w + 10
        info_lines = [
            f"Traj: {traj['dir'].name}",
            f"Steps: {meta.get('num_steps', '?')}",
            f"Seed: {meta.get('seed', '?')}",
        ]
        if positions is not None and len(positions) > 1:
            dist = np.linalg.norm(positions[-1] - positions[0])
            info_lines.append(f"Travel: {dist:.0f}px")

        for j, line in enumerate(info_lines):
            cv2.putText(grid, line, (x_info, y + 15 + j * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    if output_path:
        cv2.imwrite(str(output_path), grid)
        print(f"Saved to: {output_path}")
    else:
        cv2.imshow("Trajectories", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def animate_trajectory(traj: dict, output_path: Path | None = None, fps: int = 5, overlay: bool = True):
    """Animate a trajectory as a video or GIF-like display."""
    images = traj["images"]
    positions = traj["positions"]
    meta = traj["meta"]

    if not images:
        print("No images to animate")
        return

    img_h, img_w = images[0].shape[:2]

    print(f"Animating trajectory: {meta.get('task', 'unknown')} | {len(images)} frames")
    print("Press 'q' to quit, any other key to pause/resume")

    # Build path overlay for each frame
    if positions is not None:
        pos = positions.copy()
        pos_min = pos.min(axis=0)
        pos_max = pos.max(axis=0)
        pos_range = pos_max - pos_min
        if pos_range.max() > 0:
            scale = img_w * 0.8 / max(pos_range.max(), 1)
            offset = np.array([img_w / 2, img_h / 2]) - (pos.mean(axis=0) - pos_min) * scale
            scaled_pos = (pos - pos_min) * scale + offset
        else:
            scaled_pos = None
    else:
        scaled_pos = None

    if output_path:
        # Save as video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (img_w, img_h))

    for i, img in enumerate(images):
        display = img.copy()

        if overlay:
            # Draw trajectory path up to current frame
            if scaled_pos is not None:
                for j in range(1, i + 1):
                    pt1 = tuple(scaled_pos[j - 1].astype(int))
                    pt2 = tuple(scaled_pos[j].astype(int))
                    ratio = j / len(scaled_pos)
                    color = (0, int(255 * (1 - ratio)), int(255 * ratio))
                    cv2.line(display, pt1, pt2, color, 1)

                # Current position marker
                cv2.circle(display, tuple(scaled_pos[i].astype(int)), 5, (0, 255, 255), -1)

            # Frame counter
            cv2.putText(display, f"Frame {i}/{len(images)-1}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if output_path:
            out.write(display)
        else:
            cv2.imshow("Trajectory Animation", display)
            key = cv2.waitKey(1000 // fps)
            if key == ord('q'):
                break

    if output_path:
        out.release()
        print(f"Saved video to: {output_path}")
    else:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Visualize generated trajectories.")
    parser.add_argument("--traj-dir", type=str, default=None,
                        help="Path to a single trajectory directory")
    parser.add_argument("--task-dir", type=str, default=None,
                        help="Path to task directory containing multiple trajectories")
    parser.add_argument("--num-show", type=int, default=6,
                        help="Number of trajectories to show in grid view")
    parser.add_argument("--animate", action="store_true",
                        help="Animate a single trajectory")
    parser.add_argument("--fps", type=int, default=5,
                        help="Animation FPS")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image/video path")
    parser.add_argument("--frame-size", type=int, default=224,
                        help="Frame size for visualization (should match simulator image_size)")
    parser.add_argument("--no-overlay", action="store_true",
                        help="Disable colored trajectory overlay (show raw frames only)")
    args = parser.parse_args()

    if args.traj_dir:
        traj_dir = Path(args.traj_dir)
        traj = load_trajectory(traj_dir)

        if args.animate:
            output = Path(args.output) if args.output else None
            animate_trajectory(traj, output, args.fps, overlay=not args.no_overlay)
        else:
            output = Path(args.output) if args.output else Path("data/traj_visualization.png")
            visualize_single_trajectory(traj, output, args.frame_size)

    elif args.task_dir:
        task_dir = Path(args.task_dir)
        if args.animate:
            # Pick first trajectory from task dir for animation
            traj_dirs = sorted([d for d in task_dir.iterdir() if d.is_dir() and d.name.startswith("traj_")])
            if not traj_dirs:
                print(f"No trajectories found in {task_dir}")
                return
            traj = load_trajectory(traj_dirs[0])
            output = Path(args.output) if args.output else None
            animate_trajectory(traj, output, args.fps, overlay=not args.no_overlay)
        else:
            output = Path(args.output) if args.output else Path("data/traj_grid.png")
            visualize_trajectory_grid(task_dir, args.num_show, output, args.frame_size)

    else:
        # Default: show from data/trajectories/sperm_head
        default_dir = Path("data/trajectories/sperm_head")
        if default_dir.exists():
            print(f"No path specified, using default: {default_dir}")
            output = Path(args.output) if args.output else Path("data/traj_grid.png")
            visualize_trajectory_grid(default_dir, args.num_show, output, args.frame_size)
        else:
            print("Please specify --traj-dir or --task-dir")
            print("Example:")
            print("  python scripts/visualize_trajectories.py --task-dir data/trajectories/sperm_head")
            print("  python scripts/visualize_trajectories.py --traj-dir data/trajectories/sperm_head/traj_00000")


if __name__ == "__main__":
    main()
