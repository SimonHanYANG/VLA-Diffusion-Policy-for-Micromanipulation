"""Interactive UI to view generated trajectory animations.

Usage:
  python scripts/visualize_trajectory_ui.py
  python scripts/visualize_trajectory_ui.py --task-dir data/trajectories/embryo

Controls:
  Space / Right Arrow : Next frame
  Left Arrow          : Previous frame
  N                   : Next trajectory
  P                   : Previous trajectory
  A                   : Auto-play / Pause
  R                   : Reset to first frame
  1-5                 : Switch task (embryo/oocyte/real_sperm_head/whole_sperm/microsphere)
  Q / Esc             : Quit
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import cv2
import numpy as np


TASKS = ["embryo", "oocyte", "real_sperm_head", "whole_sperm", "microsphere"]

TASK_COLORS = {
    "embryo": (0, 200, 0),
    "oocyte": (0, 200, 200),
    "real_sperm_head": (200, 0, 200),
    "whole_sperm": (200, 200, 0),
    "microsphere": (200, 100, 0),
}


def load_trajectory_frames(traj_dir: Path, target_size: int = 640) -> list[np.ndarray]:
    """Load all frames from a trajectory directory."""
    images_dir = traj_dir / "images"
    if not images_dir.exists():
        return []

    frame_files = sorted(images_dir.glob("*.jpg"))
    frames = []
    for f in frame_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        if h != target_size or w != target_size:
            img = cv2.resize(img, (target_size, target_size))
        frames.append(img)
    return frames


def load_trajectory_meta(traj_dir: Path) -> dict:
    """Load trajectory metadata."""
    import json
    meta_file = traj_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, "r") as f:
            return json.load(f)
    return {}


def draw_info_panel(frame: np.ndarray, info: dict) -> np.ndarray:
    """Draw info overlay on frame."""
    h, w = frame.shape[:2]
    panel = frame.copy()

    # Semi-transparent background for text
    overlay = panel.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, panel, 0.4, 0, panel)

    y = 20
    color = (220, 220, 220)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1

    # Task name
    task = info.get("task", "unknown")
    cv2.putText(panel, f"Task: {task}", (10, y), font, scale, color, thickness)

    # Trajectory index
    traj_idx = info.get("traj_idx", 0)
    total_trajs = info.get("total_trajs", 0)
    cv2.putText(panel, f"Trajectory: {traj_idx}/{total_trajs}", (200, y), font, scale, color, thickness)

    # Frame counter
    frame_idx = info.get("frame_idx", 0)
    total_frames = info.get("total_frames", 0)
    cv2.putText(panel, f"Frame: {frame_idx}/{total_frames}", (450, y), font, scale, color, thickness)

    y += 25
    # Steps info
    num_steps = info.get("num_steps", 0)
    cv2.putText(panel, f"Steps: {num_steps}", (10, y), font, scale, color, thickness)

    # Auto-play status
    if info.get("auto_play", False):
        cv2.putText(panel, "AUTO", (200, y), font, scale, (0, 200, 0), thickness)

    # Controls hint
    y += 25
    cv2.putText(panel, "Space:Next  Left:Prev  N:NextTraj  P:PrevTraj  A:Auto  1-5:Task  Q:Quit",
                (10, y), font, 0.35, (150, 150, 150), thickness)

    return panel


def draw_progress_bar(frame: np.ndarray, current: int, total: int) -> np.ndarray:
    """Draw a progress bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 4
    bar_y = h - bar_h

    # Background
    cv2.rectangle(frame, (0, bar_y), (w, h), (50, 50, 50), -1)

    # Progress
    if total > 0:
        bar_w = int(w * current / total)
        cv2.rectangle(frame, (0, bar_y), (bar_w, h), (0, 200, 0), -1)

    return frame


def main():
    parser = argparse.ArgumentParser(description="Interactive trajectory viewer.")
    parser.add_argument("--task-dir", type=str, default=None,
                        help="Path to task directory (e.g., data/trajectories/embryo)")
    parser.add_argument("--task", type=str, default=None,
                        choices=TASKS, help="Task name")
    parser.add_argument("--frame-size", type=int, default=640,
                        help="Display frame size")
    parser.add_argument("--fps", type=int, default=5,
                        help="Auto-play FPS")
    args = parser.parse_args()

    base_dir = Path("data/trajectories")

    # Determine initial task
    if args.task_dir:
        current_task_dir = Path(args.task_dir)
        current_task = current_task_dir.name
    elif args.task:
        current_task = args.task
        current_task_dir = base_dir / current_task
    else:
        current_task = TASKS[0]
        current_task_dir = base_dir / current_task

    if not current_task_dir.exists():
        print(f"Task directory not found: {current_task_dir}")
        print(f"Available tasks in {base_dir}:")
        if base_dir.exists():
            for d in sorted(base_dir.iterdir()):
                if d.is_dir():
                    print(f"  {d.name}")
        return

    # Load all trajectories for current task
    traj_dirs = sorted([d for d in current_task_dir.iterdir()
                        if d.is_dir() and d.name.startswith("traj_")])
    if not traj_dirs:
        print(f"No trajectories found in {current_task_dir}")
        return

    # State
    traj_idx = 0
    frame_idx = 0
    auto_play = False
    window_name = "Trajectory Viewer"
    display_size = args.frame_size

    def load_current_trajectory():
        nonlocal frames, meta, frame_idx
        frames = load_trajectory_frames(traj_dirs[traj_idx], display_size)
        meta = load_trajectory_meta(traj_dirs[traj_idx])
        frame_idx = 0

    frames = []
    meta = {}
    load_current_trajectory()

    if not frames:
        print(f"No frames found in {traj_dirs[0]}")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    delay = int(1000 / args.fps)

    while True:
        if not frames:
            break

        # Get current frame
        frame = frames[frame_idx].copy()

        # Draw info
        info = {
            "task": meta.get("task", current_task),
            "traj_idx": traj_idx + 1,
            "total_trajs": len(traj_dirs),
            "frame_idx": frame_idx + 1,
            "total_frames": len(frames),
            "num_steps": meta.get("num_steps", len(frames) - 1),
            "auto_play": auto_play,
        }
        frame = draw_info_panel(frame, info)
        frame = draw_progress_bar(frame, frame_idx, len(frames) - 1)

        # Show
        cv2.imshow(window_name, frame)

        # Wait for key
        if auto_play:
            key = cv2.waitKey(delay) & 0xFF
        else:
            key = cv2.waitKey(0) & 0xFF

        # Handle key
        if key == ord('q') or key == 27:  # Q or Esc
            break
        elif key == ord(' ') or key == 83:  # Space or Right arrow
            frame_idx = min(frame_idx + 1, len(frames) - 1)
        elif key == 81:  # Left arrow
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord('n'):  # Next trajectory
            traj_idx = (traj_idx + 1) % len(traj_dirs)
            load_current_trajectory()
        elif key == ord('p'):  # Previous trajectory
            traj_idx = (traj_idx - 1) % len(traj_dirs)
            load_current_trajectory()
        elif key == ord('a'):  # Auto-play toggle
            auto_play = not auto_play
        elif key == ord('r'):  # Reset
            frame_idx = 0
        elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
            # Switch task
            task_idx = key - ord('1')
            if task_idx < len(TASKS):
                new_task = TASKS[task_idx]
                new_task_dir = base_dir / new_task
                if new_task_dir.exists():
                    current_task = new_task
                    current_task_dir = new_task_dir
                    traj_dirs = sorted([d for d in current_task_dir.iterdir()
                                        if d.is_dir() and d.name.startswith("traj_")])
                    if traj_dirs:
                        traj_idx = 0
                        load_current_trajectory()
        elif auto_play:
            # In auto-play, advance frame
            if frame_idx < len(frames) - 1:
                frame_idx += 1
            else:
                # Loop or go to next trajectory
                frame_idx = 0
                traj_idx = (traj_idx + 1) % len(traj_dirs)
                load_current_trajectory()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
