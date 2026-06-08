import time
from typing import Optional

import cv2
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.simulator.environment import MicroscopeEnvironment
from src.simulator.expert_generator import PIDController
from src.simulator.targets import (
    SpermHeadGenerator,
    SpermTailGenerator,
    MicrosphereGenerator,
    YeastGenerator,
    TargetGenerator,
    create_target_generator,
)
from src.utils.config import SimulatorConfig


class SimulatorViewer:
    """Real-time OpenCV-based simulator visualization.

    Keyboard controls:
      Arrow keys : manual stage movement (5 px/step)
      R          : reset environment
      A          : toggle auto-expert (PID) mode
      S          : single step in expert mode
      1-4        : switch task type
      Q / ESC    : quit
    """

    TASK_TYPES = ["microsphere", "yeast", "sperm_head", "sperm_tail"]

    def __init__(
        self,
        config: SimulatorConfig,
        task_idx: int = 0,
    ):
        self.config = config
        self.task_idx = task_idx
        self.env: Optional[MicroscopeEnvironment] = None
        self.pid: Optional[PIDController] = None
        self.obs: Optional[np.ndarray] = None
        self.info: dict = {}
        self.auto_mode = False
        self.paused = False
        self.manual_step_size = 5.0
        self.trajectory_points: list[tuple] = []
        self.window_name = "VLA Microscope Simulator"

    def _create_target_generator(self, task_name: str) -> TargetGenerator:
        generators = {
            "microsphere": lambda: MicrosphereGenerator(),
            "yeast": lambda: YeastGenerator(),
            "sperm_head": lambda: SpermHeadGenerator(),
            "sperm_tail": lambda: SpermTailGenerator(),
        }
        return generators[task_name]()

    def _reinit_env(self) -> None:
        task_name = self.TASK_TYPES[self.task_idx]
        target_gen = self._create_target_generator(task_name)
        self.env = MicroscopeEnvironment(
            config=self.config,
            target_generator=target_gen,
        )
        self.pid = PIDController(kp=0.5, ki=0.01, kd=0.1)
        self.obs, self.info = self.env.reset()
        self.trajectory_points = []

    def run(self) -> None:
        """Main visualization loop."""
        self._reinit_env()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 448, 448)

        print("=" * 60)
        print("VLA Microscope Simulator - Controls:")
        print("  Arrow keys : move stage (5 px/step)")
        print("  R          : reset environment")
        print("  A          : toggle auto PID mode")
        print("  S          : single PID step")
        print("  1-4        : switch task type")
        print("  Q / ESC    : quit")
        print("=" * 60)

        last_time = time.time()

        while True:
            display = self._compose_display()

            cv2.imshow(self.window_name, display)
            key = cv2.waitKey(10) & 0xFF

            if key == ord("q") or key == 27:  # ESC
                break

            self._handle_key(key)

            # Auto mode
            if self.auto_mode and not self.paused:
                current_time = time.time()
                if current_time - last_time >= 0.1:  # 10 Hz
                    self._pid_step()
                    last_time = current_time

        cv2.destroyAllWindows()

    def _handle_key(self, key: int) -> None:
        if self.env is None:
            return

        dx, dy = 0.0, 0.0

        if key == ord("r"):
            self.obs, self.info = self.env.reset()
            if self.pid:
                self.pid.reset()
            self.trajectory_points = []
            self.paused = False
            self.auto_mode = False
            return

        elif key == ord("a"):
            self.auto_mode = not self.auto_mode
            self.paused = False
            status = "ON" if self.auto_mode else "OFF"
            print(f"Auto PID mode: {status}")
            if self.pid:
                self.pid.reset()
            return

        elif key == ord("s"):
            if not self.auto_mode:
                self._pid_step()
            return

        elif key == ord(" "):
            self.paused = not self.paused
            return

        elif ord("1") <= key <= ord("4"):
            self.task_idx = key - ord("1")
            print(f"Switched to task: {self.TASK_TYPES[self.task_idx]}")
            self._reinit_env()
            return

        # Arrow keys (OpenCV uses different codes on different platforms)
        elif key == 0 or key == 255:
            # Extended key — read next byte
            pass
        elif key == 81:  # Left arrow (may vary)
            dx = -self.manual_step_size
        elif key == 83:  # Right arrow (may vary)
            dx = self.manual_step_size
        elif key == 82:  # Up arrow (may vary)
            dy = -self.manual_step_size
        elif key == 84:  # Down arrow (may vary)
            dy = self.manual_step_size

        if dx != 0.0 or dy != 0.0:
            action = np.array([dx, dy], dtype=np.float64)
            self.obs, reward, done, self.info = self.env.step(action)
            self.trajectory_points.append((
                int(self.info["target_position"][0]),
                int(self.info["target_position"][1]),
            ))
            if done:
                dist = self.info.get("distance", "?")
                print(f"Done! Final distance: {dist:.2f} px")

    def _pid_step(self) -> None:
        if self.env is None or self.info is None:
            return

        laser_pos = np.array([
            self.config.image_size[1] // 2,
            self.config.image_size[0] // 2,
        ], dtype=np.float64)

        ref_pt = self.env.get_target_reference_point()
        mask_h, mask_w = self.env.target_render.mask.shape
        target_center = self.info["target_position"] + ref_pt - np.array([mask_w / 2, mask_h / 2])

        action = self.pid.compute(laser_pos, target_center)

        # Clip action
        max_step = 25.0
        norm = np.linalg.norm(action)
        if norm > max_step:
            action = action / norm * max_step

        self.obs, reward, done, self.info = self.env.step(action)
        self.trajectory_points.append((
            int(self.info["target_position"][0]),
            int(self.info["target_position"][1]),
        ))

        if done:
            dist = self.info.get("distance", "?")
            print(f"PID reached target! Final distance: {dist:.2f} px")
            self.auto_mode = False

    def _compose_display(self) -> np.ndarray:
        """Build the full display with info overlay."""
        if self.obs is None:
            return np.zeros((300, 300, 3), dtype=np.uint8)

        h, w = self.obs.shape[:2]
        info_bar_height = 80
        display = np.zeros((h + info_bar_height, w, 3), dtype=np.uint8)

        # Main image
        display[:h, :w] = self.obs

        # Trajectory overlay
        if len(self.trajectory_points) > 1:
            for i in range(1, len(self.trajectory_points)):
                cv2.line(display, self.trajectory_points[i-1], self.trajectory_points[i],
                         (0, 255, 0), 1)

        # Target info line
        if self.env is not None and self.env.target_render is not None:
            ref = self.env.target_render.reference_point
            tp = self.info.get("target_position", np.zeros(2))
            mask_shape = self.env.target_render.mask.shape
            tc = (int(tp[0] + ref[0] - mask_shape[1]/2), int(tp[1] + ref[1] - mask_shape[0]/2))
            cv2.drawMarker(display, tc, (0, 255, 0), cv2.MARKER_CROSS, 10, 1)

        # Laser dot crosshair
        lx, ly = self.config.image_size[1] // 2, self.config.image_size[0] // 2
        cv2.drawMarker(display, (lx, ly), (0, 0, 255), cv2.MARKER_CROSS, 10, 1)

        # Info bar
        info_bar = display[h:, :]
        info_bar[:] = (40, 40, 40)

        lines = [
            f"Task: {self.TASK_TYPES[self.task_idx]}",
            f"Step: {self.info.get('step', 0)}",
            f"Dist: {self.info.get('distance', 0):.1f} px",
            f"Auto: {'ON' if self.auto_mode else 'OFF'}",
            f"Stage: ({self.info.get('stage_position', (0,0))[0]:.1f}, {self.info.get('stage_position', (0,0))[1]:.1f}) um",
        ]

        font = cv2.FONT_HERSHEY_SIMPLEX
        for i, line in enumerate(lines):
            cv2.putText(info_bar, line, (10, 18 + i * 16), font, 0.45, (200, 200, 200), 1)

        return display


def run_viewer():
    """Entry point for the simulator viewer."""
    config = SimulatorConfig()
    viewer = SimulatorViewer(config)
    viewer.run()


if __name__ == "__main__":
    run_viewer()
