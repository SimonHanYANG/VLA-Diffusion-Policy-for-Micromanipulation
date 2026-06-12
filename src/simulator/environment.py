from typing import Optional, Tuple

import cv2
import numpy as np
from pathlib import Path

from src.simulator.background import BackgroundGenerator, MultiImageBackground, PerlinNoiseBackground, SingleImageBackground
from src.simulator.noise import NoiseApplicator, NoiseConfig
from src.simulator.renderer import MicroscopeRenderer
from src.simulator.stage import StageSimulator
from src.simulator.targets import TargetGenerator, TargetRender
from src.utils.config import DomainRandConfig, OpticsConfig, SimulatorConfig


class MicroscopeEnvironment:
    """Gym-like environment for simulated microscope navigation.

    The stage moves in micrometers (matching the real Nikon Ti2E API).
    Object translation in the image is the reverse of stage movement
    (moving stage right makes the object appear to move left in the image),
    with a +/-10% random scale factor per episode to simulate calibration error.
    """

    def __init__(
        self,
        config: SimulatorConfig,
        target_generator: TargetGenerator,
        background: BackgroundGenerator | None = None,
        stage: StageSimulator | None = None,
    ):
        self.config = config
        self.target_generator = target_generator

        # 根据配置创建背景生成器
        if background is not None:
            self.background = background
        elif config.background_type == "multi_image":
            self.background = MultiImageBackground(
                image_paths=config.background_images,
                brightness_range=config.background_brightness_range,
            )
        elif config.background_type == "single_image":
            self.background = SingleImageBackground(
                image_path=Path(config.background_image),
                brightness_range=config.background_brightness_range,
            )
        else:
            self.background = PerlinNoiseBackground()

        self.stage = stage or StageSimulator()
        self.stage.connect()

        noise_applicator = NoiseApplicator(config.noise) if config.noise else None
        self.renderer = MicroscopeRenderer(
            image_size=config.image_size,
            background=self.background,
            noise=noise_applicator,
            domain_rand=config.domain_randomization,
            optics=config.optics,  # Pass None directly, don't fallback to OpticsConfig()
            blend_factor_range=config.blend_factor_range,
            edge_fade_ratio=config.edge_fade_ratio,
            brightness_adapt_range=config.brightness_adapt_range,
            edge_blur_kernel_range=config.edge_blur_kernel_range,
            edge_blur_sigma_range=config.edge_blur_sigma_range,
        )

        # State
        self.target_render: Optional[TargetRender] = None
        self.target_position: Optional[np.ndarray] = None
        self.laser_position: np.ndarray = np.array([
            config.image_size[1] // 2, config.image_size[0] // 2
        ])
        self.rng: Optional[np.random.Generator] = None
        self.step_count: int = 0
        self.scale_factor: float = 1.0
        self._current_image: Optional[np.ndarray] = None

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """Generate random target at random start position, render first frame."""
        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.stage.reset_position()

        # Generate target
        self.target_render = self.target_generator.generate(self.config.image_size, self.rng)

        # Random start position from laser dot (scaled for resolution)
        angle = self.rng.uniform(0, 2 * np.pi)
        distance = self.rng.uniform(self.config.scaled_start_distance_min, self.config.scaled_start_distance_max)
        start_x = self.laser_position[0] + distance * np.cos(angle)
        start_y = self.laser_position[1] + distance * np.sin(angle)
        h, w = self.config.image_size
        margin = int(20 * self.config.scale_factor)
        start_x = np.clip(start_x, margin, w - margin)
        start_y = np.clip(start_y, margin, h - margin)
        self.target_position = np.array([start_x, start_y], dtype=np.float64)

        # Random scale factor with +/-10% noise to simulate calibration error
        self.scale_factor = self.config.um_per_pixel * (1.0 + self.rng.uniform(
            -self.config.um_per_pixel_noise, self.config.um_per_pixel_noise
        ))

        # Render first frame
        self._current_image = self.renderer.render(
            self.target_render.mask, (self.target_position[0], self.target_position[1]), self.rng,
            target_texture=self.target_render.texture,
        )

        info = self._get_info()
        return self._current_image.copy(), info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute action and return (obs, reward, done, info).

        Args:
            action: (dx, dy) desired object movement in pixel space.

        Returns:
            (image, reward, done, info)
        """
        dx, dy = float(action[0]), float(action[1])

        # Object moves opposite to stage movement
        self.target_position[0] += dx
        self.target_position[1] += dy

        # Also update stage position (for logging / sim-to-real consistency)
        stage_dx = -dx * self.scale_factor
        stage_dy = -dy * self.scale_factor
        self.stage.move_xy_relative(stage_dx, stage_dy)

        self.step_count += 1

        # Render
        self._current_image = self.renderer.render(
            self.target_render.mask, (self.target_position[0], self.target_position[1]), self.rng,
            target_texture=self.target_render.texture,
        )

        # Reward: negative distance
        distance = self._compute_distance()
        reward = -distance

        # Done conditions
        done = distance < self.config.scaled_success_tolerance
        if self.step_count >= self.config.max_steps_per_episode:
            done = True

        info = self._get_info()
        return self._current_image.copy(), reward, done, info

    def render(self) -> np.ndarray:
        """Return current rendered image."""
        if self._current_image is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._current_image.copy()

    def get_target_reference_point(self) -> np.ndarray:
        """Return the target's reference point in image coordinates."""
        if self.target_render is None:
            raise RuntimeError("Environment not initialized.")
        return np.array(self.target_render.reference_point, dtype=np.float64)

    def _compute_distance(self) -> float:
        """Euclidean distance from target reference point to laser dot."""
        if self.target_render is None:
            return float("inf")
        ref = np.array(self.target_render.reference_point, dtype=np.float64)
        mask_h, mask_w = self.target_render.mask.shape
        mask_center = np.array([mask_w / 2, mask_h / 2], dtype=np.float64)
        return float(np.linalg.norm(self.target_position + ref - mask_center - self.laser_position))

    def _get_info(self) -> dict:
        return {
            "step": self.step_count,
            "target_position": self.target_position.copy(),
            "laser_position": self.laser_position.copy(),
            "distance": self._compute_distance(),
            "scale_factor": self.scale_factor,
            "stage_position": self.stage.get_xy_position(),
            "target_label": self.target_render.label if self.target_render else "",
        }
