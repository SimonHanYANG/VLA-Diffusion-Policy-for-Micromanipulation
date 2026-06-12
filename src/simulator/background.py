from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

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


class SingleImageBackground(BackgroundGenerator):
    """使用单张真实显微镜背景图片。

    从指定路径加载一张背景图片，支持随机裁剪、翻转和亮度调整。
    图片只加载一次并缓存，后续调用直接使用缓存。
    """

    def __init__(
        self,
        image_path: Path = Path("data/backgrounds_test/bg_00000.png"),
        brightness_range: Tuple[float, float] = (0.8, 1.2),
    ):
        self.image_path = Path(image_path)
        self.brightness_range = brightness_range
        self._cached_image: np.ndarray | None = None

    def _load_image(self) -> np.ndarray:
        """加载并缓存背景图片。"""
        if self._cached_image is None:
            img = cv2.imread(str(self.image_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"无法加载背景图片: {self.image_path}")
            self._cached_image = img
        return self._cached_image

    def generate(self, size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        w, h = size
        img = self._load_image()
        ih, iw = img.shape

        # 随机裁剪
        if iw > w and ih > h:
            x = rng.integers(0, iw - w + 1)
            y = rng.integers(0, ih - h + 1)
            crop = img[y:y+h, x:x+w].copy()
        else:
            # 如果图片太小，进行填充
            pad_h = max(0, h - ih)
            pad_w = max(0, w - iw)
            crop = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
            crop = crop[:h, :w].copy()

        # 随机翻转
        if rng.random() < 0.5:
            crop = np.fliplr(crop)
        if rng.random() < 0.5:
            crop = np.flipud(crop)

        # 亮度调整
        brightness = rng.uniform(*self.brightness_range)
        crop = np.clip(crop.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        return crop


class MultiImageBackground(BackgroundGenerator):
    """从多张指定背景图中随机选取，支持亮度微调。

    专门用于从 backgrounds_test 中选取 bg_00001~00003 作为背景。
    所有操作保持灰度，不引入彩色。
    """

    def __init__(
        self,
        image_paths: List[str | Path],
        brightness_range: Tuple[float, float] = (0.85, 1.15),
        blur_sigma_max: float = 0.5,
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.brightness_range = brightness_range
        self.blur_sigma_max = blur_sigma_max
        self._cached_images: dict[int, np.ndarray] = {}

    def _load_image(self, idx: int) -> np.ndarray:
        """加载并缓存指定索引的背景图片。"""
        if idx not in self._cached_images:
            path = self.image_paths[idx]
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"无法加载背景图片: {path}")
            self._cached_images[idx] = img
        return self._cached_images[idx]

    def generate(self, size: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
        w, h = size

        # 1. 随机选一张背景
        idx = rng.integers(0, len(self.image_paths))
        img = self._load_image(idx)
        ih, iw = img.shape

        # 2. 随机裁剪到目标尺寸
        if iw > w and ih > h:
            x = rng.integers(0, iw - w + 1)
            y = rng.integers(0, ih - h + 1)
            crop = img[y:y+h, x:x+w].copy()
        else:
            # 图片太小时反射填充
            pad_h = max(0, h - ih)
            pad_w = max(0, w - iw)
            crop = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
            crop = crop[:h, :w].copy()

        # 3. 随机翻转增加多样性
        if rng.random() < 0.5:
            crop = np.fliplr(crop)
        if rng.random() < 0.5:
            crop = np.flipud(crop)

        # 4. 亮度微调（只调整明暗，不加颜色）
        brightness = rng.uniform(*self.brightness_range)
        crop = np.clip(crop.astype(np.float32) * brightness, 0, 255).astype(np.uint8)

        # 5. 可选轻微高斯模糊（模拟焦距微调）
        if self.blur_sigma_max > 0 and rng.random() < 0.3:
            sigma = rng.uniform(0.3, self.blur_sigma_max)
            ksize = int(sigma * 6) | 1  # 确保奇数
            ksize = max(ksize, 3)
            crop = cv2.GaussianBlur(crop, (ksize, ksize), sigma)

        return crop


class BackgroundGeneratorFactory:
    """Creates the appropriate background generator from config."""

    @staticmethod
    def create(background_type: str = "perlin", image_dir: Path | None = None,
               image_path: Path | None = None,
               image_paths: List[str | Path] | None = None,
               **kwargs) -> BackgroundGenerator:
        if background_type == "multi_image" and image_paths is not None:
            return MultiImageBackground(image_paths=image_paths, **kwargs)
        if background_type == "single_image" and image_path is not None:
            return SingleImageBackground(image_path=image_path, **kwargs)
        if background_type == "image" and image_dir is not None:
            return ImageBackground(image_dir)
        return PerlinNoiseBackground(**kwargs)
