import cv2
import numpy as np

from src.utils.config import NoiseConfig


class NoiseApplicator:
    """Applies noise models to a rendered microscope image."""

    def __init__(self, config: NoiseConfig = NoiseConfig()):
        self.config = config

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply all noise models in sequence. Input/output in [0, 255] uint8.

        Order: Gaussian -> salt-pepper -> flicker -> motion blur.
        """
        img = image.astype(np.float32)

        # 1. Gaussian noise
        if self.config.gaussian_std > 0:
            noise = rng.normal(0, self.config.gaussian_std, img.shape).astype(np.float32)
            img += noise

        img = np.clip(img, 0, 255).astype(np.uint8)

        # 2. Salt-and-pepper noise
        if self.config.salt_pepper_prob > 0 and rng.random() < 0.5:
            rand = rng.random(img.shape[:2])
            img[rand < self.config.salt_pepper_prob / 2] = 0
            img[rand > 1 - self.config.salt_pepper_prob / 2] = 255

        # 3. Illumination flicker (multiplicative)
        if rng.random() < self.config.flicker_prob:
            factor = rng.uniform(self.config.flicker_intensity_min, self.config.flicker_intensity_max)
            img = np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

        # 4. Motion blur
        if rng.random() < self.config.motion_blur_prob:
            kernel_size = rng.integers(3, self.config.motion_blur_kernel_max + 1)
            if kernel_size % 2 == 0:
                kernel_size += 1
            angle = rng.uniform(0, np.pi)
            img = self._motion_blur(img, kernel_size, angle)

        return img

    def _motion_blur(self, image: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
        """Apply directional motion blur via 2D kernel convolution."""
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = kernel_size // 2
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for i in range(kernel_size):
            dx = (i - center) * cos_a
            dy = (i - center) * sin_a
            x = int(round(center + dx))
            y = int(round(center + dy))
            if 0 <= x < kernel_size and 0 <= y < kernel_size:
                kernel[y, x] = 1.0
        kernel /= kernel.sum() + 1e-8
        return cv2.filter2D(image, -1, kernel)
