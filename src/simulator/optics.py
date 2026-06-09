"""Optical effects post-processing for realistic microscope simulation.

Applies vignetting, point spread function (PSF) convolution, and
chromatic aberration to mimic real microscope optics.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils.config import OpticsConfig


class OpticsProcessor:
    """Applies optical effects to microscope images.

    Effects (applied in order):
    1. Vignetting — radial darkening from image center
    2. PSF convolution — slight Gaussian blur mimicking microscope optics
    3. Chromatic aberration — subtle radial RGB channel shift
    """

    def __init__(self, config: OpticsConfig = OpticsConfig()):
        self.config = config
        self._vignette_cache: dict[tuple, np.ndarray] = {}

    def apply(self, image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Apply all optical effects. Input/output BGR uint8."""
        img = image.copy()

        # 1. Vignetting
        if rng.random() < self.config.vignette_prob:
            strength = rng.uniform(*self.config.vignette_strength_range)
            img = self._apply_vignetting(img, strength)

        # 2. PSF convolution (optical blur)
        if rng.random() < self.config.psf_prob:
            sigma = rng.uniform(*self.config.psf_sigma_range)
            img = self._apply_psf(img, sigma)

        # 3. Chromatic aberration
        if rng.random() < self.config.chromatic_prob:
            shift = rng.uniform(*self.config.chromatic_shift_range)
            img = self._apply_chromatic_aberration(img, shift)

        return img

    def _apply_vignetting(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply radial darkening from image center.

        Vignette factor: 1.0 - strength * (r/r_max)^power
        where r is distance from center, r_max is corner distance.
        """
        h, w = image.shape[:2]
        cache_key = (h, w, round(strength, 3))

        if cache_key in self._vignette_cache:
            vignette = self._vignette_cache[cache_key]
        else:
            cy, cx = h / 2.0, w / 2.0
            y_grid, x_grid = np.ogrid[:h, :w]
            r = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
            r_max = np.sqrt(cx ** 2 + cy ** 2)
            r_norm = r / r_max

            # Quadratic vignetting model
            power = self.config.vignette_power
            vignette = 1.0 - strength * np.power(r_norm, power)
            vignette = np.clip(vignette, 0.0, 1.0)
            self._vignette_cache[cache_key] = vignette

        # Apply as multiplicative factor
        img_float = image.astype(np.float32)
        for c in range(3):
            img_float[:, :, c] *= vignette
        return np.clip(img_float, 0, 255).astype(np.uint8)

    def _apply_psf(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """Apply Gaussian PSF convolution (simulates microscope optical blur)."""
        ksize = int(np.ceil(sigma * 6)) | 1
        ksize = max(3, ksize)
        return cv2.GaussianBlur(image, (ksize, ksize), sigma)

    def _apply_chromatic_aberration(self, image: np.ndarray, max_shift: float) -> np.ndarray:
        """Apply subtle radial chromatic aberration.

        Shifts red and blue channels outward from center by a small amount.
        Green channel stays in place (reference).
        """
        h, w = image.shape[:2]
        cy, cx = h / 2.0, w / 2.0

        y_grid, x_grid = np.ogrid[:h, :w]
        dx = x_grid - cx
        dy = y_grid - cy
        r = np.sqrt(dx ** 2 + dy ** 2)
        r_max = np.sqrt(cx ** 2 + cy ** 2)
        r_norm = r / r_max + 1e-8  # avoid division by zero

        # Radial unit vectors
        ux = dx / r_norm
        uy = dy / r_norm

        # Shift maps for red (outward) and blue (inward) channels
        shift_red = max_shift * r_norm
        shift_blue = -max_shift * 0.5 * r_norm  # blue shifts less

        # Build remap matrices
        map_x_red = (x_grid + ux * shift_red).astype(np.float32)
        map_y_red = (y_grid + uy * shift_red).astype(np.float32)
        map_x_blue = (x_grid + ux * shift_blue).astype(np.float32)
        map_y_blue = (y_grid + uy * shift_blue).astype(np.float32)

        result = image.copy()
        result[:, :, 2] = cv2.remap(image[:, :, 2], map_x_red, map_y_red, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)  # Red channel (BGR index 2)
        result[:, :, 0] = cv2.remap(image[:, :, 0], map_x_blue, map_y_blue, cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REFLECT)  # Blue channel (BGR index 0)
        return result
