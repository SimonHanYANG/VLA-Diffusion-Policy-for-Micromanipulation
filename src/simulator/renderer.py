from typing import Tuple

import cv2
import numpy as np

from src.simulator.background import BackgroundGenerator
from src.simulator.noise import NoiseApplicator, NoiseConfig
from src.utils.config import DomainRandConfig


class MicroscopeRenderer:
    """Composes the final microscope image.

    Composition order:
    1. Generate background
    2. Paste target object at position
    3. Draw red laser dot at image center
    4. Apply domain randomization (brightness, contrast, blur)
    5. Apply noise models
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        background: BackgroundGenerator,
        noise: NoiseApplicator | None = None,
        domain_rand: DomainRandConfig = DomainRandConfig(),
    ):
        self.image_size = image_size
        self.background = background
        self.noise = noise or NoiseApplicator()
        self.domain_rand = domain_rand
        self.laser_position = (image_size[1] // 2, image_size[0] // 2)
        self.laser_radius = 3
        self.laser_color_bgr = (0, 0, 255)  # Red in BGR

    def set_laser_position(self, x: int, y: int) -> None:
        self.laser_position = (x, y)

    def render(
        self,
        target_mask: np.ndarray,
        target_position: Tuple[float, float],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Render a complete microscope image.

        Args:
            target_mask: (H, W) uint8 mask of the target (255=target).
            target_position: (x, y) center position of the target in the rendered image.
            rng: random generator for noise and domain randomization.

        Returns:
            (H, W, 3) BGR uint8 image.
        """
        h, w = self.image_size

        # 1. Background
        bg = self.background.generate((w, h), rng)
        image = cv2.cvtColor(bg, cv2.COLOR_GRAY2BGR)

        # 2. Paste target at position
        target_h, target_w = target_mask.shape
        tx = int(round(target_position[0] - target_w / 2))
        ty = int(round(target_position[1] - target_h / 2))

        # Compute valid paste region
        src_x1 = max(0, -tx)
        src_y1 = max(0, -ty)
        src_x2 = min(target_w, w - tx)
        src_y2 = min(target_h, h - ty)
        dst_x1 = max(0, tx)
        dst_y1 = max(0, ty)
        dst_x2 = min(w, tx + target_w)
        dst_y2 = min(h, ty + target_h)

        if src_x2 > src_x1 and src_y2 > src_y1:
            src_region = target_mask[src_y1:src_y2, src_x1:src_x2]
            dst_region = image[dst_y1:dst_y2, dst_x1:dst_x2]

            alpha = (src_region / 255.0)[:, :, np.newaxis]
            dst_region[:] = (alpha * 200 + (1 - alpha) * dst_region).astype(np.uint8)

        # 3. Red laser dot at center
        cv2.circle(image, self.laser_position, self.laser_radius, self.laser_color_bgr, -1)

        # 4. Domain randomization
        image = self._apply_domain_randomization(image, rng)

        # 5. Noise
        image = self.noise.apply(image, rng)

        return image

    def _apply_domain_randomization(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply random brightness, contrast, and Gaussian blur."""
        if rng.random() < 0.5:
            brightness = rng.uniform(*self.domain_rand.brightness_range)
            image = np.clip(image.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        if rng.random() < 0.5:
            contrast = rng.uniform(*self.domain_rand.contrast_range)
            mean = image.astype(np.float32).mean()
            image = np.clip((image.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)

        if rng.random() < self.domain_rand.blur_prob:
            sigma = rng.uniform(0.5, self.domain_rand.blur_sigma_max)
            ksize = int(np.ceil(sigma * 6)) | 1  # Odd kernel
            ksize = max(3, ksize)
            image = cv2.GaussianBlur(image, (ksize, ksize), sigma)

        return image
