from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from src.utils.perlin import generate_perlin_noise


class BackgroundGenerator(ABC):
    @abstractmethod
    def generate(self, size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        """Return (H, W) uint8 grayscale background. ``size`` is (width, height)."""
        ...


class PerlinNoiseBackground(BackgroundGenerator):
    """Multi-octave Perlin noise as synthetic microscope background."""

    def __init__(self, base_scale: float = 80.0, octaves: int = 3,
                 persistence: float = 0.3, lacunarity: float = 2.0,
                 intensity_min: int = 140, intensity_max: int = 160):
        self.base_scale = base_scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max

    def generate(self, size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        w, h = size
        seed = rng.integers(0, 2**31)
        noise = generate_perlin_noise(
            width=w, height=h, scale=self.base_scale,
            octaves=self.octaves, persistence=self.persistence,
            lacunarity=self.lacunarity, seed=seed,
        )
        bg = (noise * (self.intensity_max - self.intensity_min) + self.intensity_min).astype(np.uint8)
        return bg


class ImageBackground(BackgroundGenerator):
    """Random crops from a directory of real microscope background images.
    Falls back to Perlin noise if no images are available."""

    def __init__(self, image_dir: Path | None = None):
        self.images: list[Path] = []
        if image_dir is not None and image_dir.exists():
            self.images = sorted(image_dir.glob("*"))
            self.images = [p for p in self.images if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
        self._fallback = PerlinNoiseBackground()

    def generate(self, size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        if not self.images:
            return self._fallback.generate(size, rng)

        img_path = rng.choice(self.images)
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return self._fallback.generate(size, rng)

        w, h = size
        ih, iw = img.shape
        if iw < w or ih < h:
            # Pad or fall back
            if iw < w:
                img = np.pad(img, ((0, 0), (0, w - iw)), mode="reflect")
                iw = w
            if ih < h:
                img = np.pad(img, ((0, h - ih), (0, 0)), mode="reflect")
                ih = h

        x = rng.integers(0, iw - w + 1) if iw > w else 0
        y = rng.integers(0, ih - h + 1) if ih > h else 0
        crop = img[y:y+h, x:x+w]

        # Random flip
        if rng.random() < 0.5:
            crop = np.fliplr(crop)
        if rng.random() < 0.5:
            crop = np.flipud(crop)

        return crop.copy()


class BackgroundGeneratorFactory:
    """Creates the appropriate background generator from config."""

    @staticmethod
    def create(background_type: str = "perlin", image_dir: Path | None = None,
               **kwargs) -> BackgroundGenerator:
        if background_type == "image" and image_dir is not None:
            return ImageBackground(image_dir)
        return PerlinNoiseBackground(**kwargs)
