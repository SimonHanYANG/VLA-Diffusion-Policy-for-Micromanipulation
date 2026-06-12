import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from tqdm import tqdm

from src.simulator.background import BackgroundGenerator, MultiImageBackground, PerlinNoiseBackground, SingleImageBackground
from src.simulator.environment import MicroscopeEnvironment
from src.simulator.targets import TargetGenerator, create_target_generator
from src.utils.config import SimulatorConfig, TaskConfig


class PIDController:
    """Discrete 2D PID controller for micro-object navigation."""

    def __init__(self, kp: float = 0.5, ki: float = 0.01, kd: float = 0.1, dt: float = 0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt
        self._integral = np.zeros(2)
        self._prev_error = np.zeros(2)

    def compute(self, setpoint: np.ndarray, current: np.ndarray) -> np.ndarray:
        """Return control signal (dx, dy) to move current toward setpoint."""
        error = setpoint - current
        self._integral += error * self.dt
        derivative = (error - self._prev_error) / self.dt
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self) -> None:
        self._integral = np.zeros(2)
        self._prev_error = np.zeros(2)


class ExpertDemonstrationGenerator:
    """Generates expert demonstration trajectories using PID control.

    For each trajectory:
    1. Reset environment (random target, random start position)
    2. While not done (distance > tolerance):
       a. Get current reference point position
       b. Compute PID action toward laser dot
       c. Add random action noise to simulate human jitter
       d. Step environment
       e. Record (image, action, position)
    3. Save trajectory to disk
    """

    def __init__(
        self,
        config: SimulatorConfig,
        target_generator: TargetGenerator,
        background: BackgroundGenerator | None = None,
        pid_kp: float = 0.5,
        pid_ki: float = 0.01,
        pid_kd: float = 0.1,
        action_noise_std: float = 0.05,
    ):
        self.config = config
        self.target_generator = target_generator
        self.background = background or PerlinNoiseBackground()
        self.pid = PIDController(kp=pid_kp, ki=pid_ki, kd=pid_kd)
        self.action_noise_std = action_noise_std

    def generate_trajectories(
        self,
        num_trajectories: int,
        output_dir: Path,
        task_name: str = "unknown",
        seed: int = 0,
        image_quality: int = 90,
    ) -> list[Path]:
        """Generate N expert demonstration trajectories."""
        output_dir.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(seed)
        saved_paths = []

        for traj_idx in tqdm(range(num_trajectories), desc=f"Generating {task_name}"):
            traj_seed = rng.integers(0, 2**31)
            traj_path = output_dir / f"traj_{traj_idx:05d}"
            traj_path.mkdir(parents=True, exist_ok=True)
            images_dir = traj_path / "images"
            images_dir.mkdir(exist_ok=True)

            result = self._run_single_trajectory(traj_seed, images_dir, image_quality)
            if result is None:
                continue

            actions, positions, meta = result

            np.save(traj_path / "actions.npy", actions)
            np.save(traj_path / "positions.npy", positions)

            meta["task"] = task_name
            meta["traj_index"] = traj_idx
            meta["seed"] = int(traj_seed)
            with open(traj_path / "meta.json", "w") as f:
                json.dump(meta, f, indent=2, default=str)

            saved_paths.append(traj_path)

        return saved_paths

    def _run_single_trajectory(
        self, seed: int, images_dir: Path, image_quality: int
    ) -> Optional[Tuple[np.ndarray, np.ndarray, dict]]:
        """Run one episode. Returns (actions, positions, meta) or None if failed."""
        import cv2

        env = MicroscopeEnvironment(
            config=self.config,
            target_generator=self.target_generator,
            background=self.background,
        )

        obs, info = env.reset(seed=seed)
        self.pid.reset()

        action_rng = np.random.default_rng(seed + 1)  # separate RNG for action noise

        images: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        positions: list[np.ndarray] = []

        images.append(obs)
        positions.append(info["target_position"].copy())

        laser_pos = np.array([self.config.image_size[1] // 2, self.config.image_size[0] // 2],
                             dtype=np.float64)

        for _ in range(self.config.max_steps_per_episode):
            # Current reference point in image
            ref_pt = env.get_target_reference_point()
            mask_h, mask_w = env.target_render.mask.shape
            target_center = info["target_position"] + ref_pt - np.array([mask_w / 2, mask_h / 2])

            # PID action toward laser
            action = self.pid.compute(laser_pos, target_center)

            # Add human-like noise
            noise = action_rng.standard_normal(2) * self.action_noise_std * np.linalg.norm(action)
            action_noisy = action + noise

            # Clip action magnitude (scaled for resolution)
            max_step = 30.0 * self.config.scale_factor
            norm = np.linalg.norm(action_noisy)
            if norm > max_step:
                action_noisy = action_noisy / norm * max_step

            obs, reward, done, info = env.step(action_noisy)

            actions.append(action_noisy.copy())
            positions.append(info["target_position"].copy())
            images.append(obs)

            if done:
                break

        if len(actions) < 2:
            return None

        # Save images as JPEG
        for i, img in enumerate(images):
            img_path = images_dir / f"{i:04d}.jpg"
            cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, image_quality])

        actions_arr = np.array(actions, dtype=np.float32)
        positions_arr = np.array(positions, dtype=np.float32)

        meta = {
            "num_steps": len(actions),
            "target_label": env.target_render.label,
            "scale_factor": float(info["scale_factor"]),
            "pid_gains": {"kp": self.pid.kp, "ki": self.pid.ki, "kd": self.pid.kd},
            "action_noise_std": self.action_noise_std,
        }

        return actions_arr, positions_arr, meta


def generate_data_for_task(
    task_config: TaskConfig,
    sim_config: SimulatorConfig,
    output_root: Path,
    num_trajectories: int = 5000,
    seed: int = 0,
    background: BackgroundGenerator | None = None,
) -> list[Path]:
    """Convenience function to generate data for a single task.

    Args:
        task_config: Task configuration.
        sim_config: Simulator configuration.
        output_root: Root output directory.
        num_trajectories: Number of trajectories to generate.
        seed: Random seed.
        background: Optional background generator. If None, creates from sim_config.
    """
    target_gen = create_target_generator(
        task_config.target_generator, task_config.target_params
    )

    # 根据配置创建背景生成器（如果未提供）
    if background is None:
        if sim_config.background_type == "multi_image":
            background = MultiImageBackground(
                image_paths=sim_config.background_images,
                brightness_range=sim_config.background_brightness_range,
            )
        elif sim_config.background_type == "single_image":
            background = SingleImageBackground(
                image_path=Path(sim_config.background_image),
                brightness_range=sim_config.background_brightness_range,
            )
        else:
            background = PerlinNoiseBackground()

    gen = ExpertDemonstrationGenerator(
        config=sim_config,
        target_generator=target_gen,
        background=background,
        pid_kp=task_config.pid_kp,
        pid_ki=task_config.pid_ki,
        pid_kd=task_config.pid_kd,
    )
    output_dir = output_root / task_config.name
    return gen.generate_trajectories(
        num_trajectories=num_trajectories,
        output_dir=output_dir,
        task_name=task_config.name,
        seed=seed,
    )
