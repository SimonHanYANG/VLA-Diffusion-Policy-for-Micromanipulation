"""Target generators for microscope simulation.

Provides:
- MicrosphereGenerator: Procedural microsphere (no real images available yet)
- RealImageTargetGenerator: Loads real cell images from pre-segmented data
- SpermTailFromImageGenerator: Extracts sperm tail tip from whole sperm images
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from src.utils.perlin import generate_perlin_noise


def _get_resolution_scale(image_size: Tuple[int, int], reference: int = 224) -> float:
    """计算分辨率相对于参考尺寸的线性缩放系数。"""
    return min(image_size) / reference


def _generate_target_texture(
    shape: Tuple[int, int],
    base_intensity: float,
    rng: np.random.Generator,
    texture_scale: float = 30.0,
    texture_strength: float = 0.15,
) -> np.ndarray:
    """Generate a textured intensity map for a target.

    Uses Perlin noise to add subtle internal texture, making the target
    look more like a real microscopy object rather than a flat gray shape.

    Args:
        shape: (H, W) of the target region.
        base_intensity: Base gray level (e.g. 70 for sperm head).
        rng: Random generator.
        texture_scale: Scale of Perlin noise features.
        texture_strength: How much texture varies (0-1).

    Returns:
        (H, W) float32 intensity map.
    """
    h, w = shape
    seed = rng.integers(0, 2**31)
    noise = generate_perlin_noise(
        width=w, height=h, scale=texture_scale,
        octaves=3, persistence=0.5, lacunarity=2.0, seed=seed,
    )
    # noise is in [0, 1], center around 0
    texture = base_intensity * (1.0 + texture_strength * (noise - 0.5) * 2)
    return np.clip(texture, 0, 255).astype(np.float32)


def _radial_gradient(shape: Tuple[int, int], center: Tuple[float, float]) -> np.ndarray:
    """Generate a radial distance map normalized to [0, 1]."""
    h, w = shape
    y_grid, x_grid = np.ogrid[:h, :w]
    r = np.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)
    r_max = max(np.sqrt(center[0] ** 2 + center[1] ** 2),
                np.sqrt((w - center[0]) ** 2 + (h - center[1]) ** 2))
    return (r / r_max).astype(np.float32)


@dataclass
class TargetRender:
    mask: np.ndarray
    reference_point: Tuple[float, float]
    label: str
    texture: np.ndarray | None = None  # (H, W) float32 intensity map, None = use default (200)


class TargetGenerator(ABC):
    @abstractmethod
    def generate(self, image_size: Tuple[int, int], rng: np.random.Generator) -> TargetRender:
        ...


class MicrosphereGenerator(TargetGenerator):
    """Rigid circle with solid or ring texture, radius 15-25 px.

    Real microspheres appear as dark circular objects with:
    - Subtle internal texture (not uniform gray)
    - Slightly brighter center highlight
    - Soft edge falloff
    - Intensity range ~80-120 (not 200)

    Note: No real microsphere images available yet. Using procedural generation.
    """

    def __init__(self, radius_min: float = 15.0, radius_max: float = 25.0,
                 ring_prob: float = 0.3, ring_thickness: float = 3.0):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.ring_prob = ring_prob
        self.ring_thickness = ring_thickness

    def generate(self, image_size: Tuple[int, int], rng: np.random.Generator) -> TargetRender:
        h, w = image_size
        scale = _get_resolution_scale(image_size)
        radius = rng.uniform(self.radius_min * scale, self.radius_max * scale)
        cx, cy = w / 2.0, h / 2.0

        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

        if rng.random() < self.ring_prob:
            thickness = self.ring_thickness * rng.uniform(0.8, 1.2) * scale
            mask = ((dist >= radius - thickness) & (dist <= radius)).astype(np.uint8) * 255
        else:
            sigma = radius * 0.05
            mask = (255 * (1.0 - np.clip((dist - radius + sigma) / (2 * sigma), 0.0, 1.0))).astype(np.uint8)
            mask[dist <= radius - sigma] = 255

        # Generate realistic texture
        base_intensity = rng.uniform(80, 120)
        texture = _generate_target_texture((h, w), base_intensity, rng, texture_scale=20.0)

        # Add center highlight (real microspheres have a bright spot)
        highlight_strength = rng.uniform(0.05, 0.15)
        highlight_sigma = radius * 0.4
        highlight = np.exp(-dist ** 2 / (2 * highlight_sigma ** 2))
        texture = texture * (1.0 + highlight_strength * highlight)

        return TargetRender(mask=mask, reference_point=(cx, cy), label="microsphere", texture=texture)


class RealImageTargetGenerator(TargetGenerator):
    """从预分割的真实显微镜图片加载目标物体。

    支持加载 individual_obj 中的胚胎、卵母细胞、精子头等图片。
    图片应为带 alpha 通道的 PNG（alpha 作为 mask）或灰度图（Otsu 阈值提取 mask）。

    自动缩放：根据目标图像尺寸自动计算缩放比例，确保物体不会太大。
    """

    def __init__(
        self,
        image_dir: str | Path,
        scale_range: Tuple[float, float] = (0.3, 0.6),
        target_occupancy: Tuple[float, float] = (0.4, 0.7),
        label: str = "real_target",
        crop_margin: int = 5,
        edge_blur_range: Tuple[float, float] = (0.8, 2.5),
    ):
        """
        Args:
            image_dir: 图片目录路径。
            scale_range: 额外随机缩放范围（在自动缩放基础上）。
            target_occupancy: 目标物体占图像的比例范围 (min, max)。
                例如 (0.4, 0.7) 表示物体占图像 40%-70% 的尺寸。
            label: 目标标签。
            crop_margin: 裁剪边距。
            edge_blur_range: 边缘模糊范围。
        """
        self.image_dir = Path(image_dir)
        self.scale_range = scale_range
        self.target_occupancy = target_occupancy
        self.label = label
        self.crop_margin = crop_margin
        self.edge_blur_range = edge_blur_range

        # 扫描目录中的所有图片
        self._image_paths: List[Path] = []
        if self.image_dir.exists():
            for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
                self._image_paths.extend(sorted(self.image_dir.glob(ext)))

        if not self.image_paths:
            raise FileNotFoundError(f"目录中没有找到图片: {self.image_dir}")

    @property
    def image_paths(self) -> List[Path]:
        return self._image_paths

    def _load_image_and_mask(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """加载图片并提取 mask。

        Returns:
            (gray_image, mask): 灰度图和二值 mask，都是 (H, W) 格式。
        """
        # 读取图片（带 alpha 通道）
        img_rgba = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img_rgba is None:
            raise ValueError(f"无法加载图片: {path}")

        if img_rgba.ndim == 3 and img_rgba.shape[2] == 4:
            # RGBA 图片：alpha 通道作为 mask
            gray = cv2.cvtColor(img_rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
            alpha = img_rgba[:, :, 3]
            mask = (alpha > 128).astype(np.uint8) * 255
        elif img_rgba.ndim == 3:
            # BGR 图片：使用 Otsu 阈值
            gray = cv2.cvtColor(img_rgba, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            # 灰度图：使用 Otsu 阈值
            gray = img_rgba
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 形态学开运算清理 mask 中的小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        return gray, mask

    def _crop_to_content(self, gray: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        """裁剪到内容的 tight bbox，保留少量边距。

        Returns:
            (cropped_gray, cropped_mask, (offset_x, offset_y))
        """
        coords = cv2.findNonZero(mask)
        if coords is None:
            return gray, mask, (0, 0)

        x, y, bw, bh = cv2.boundingRect(coords)

        # 添加边距
        margin = self.crop_margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(gray.shape[1], x + bw + margin)
        y2 = min(gray.shape[0], y + bh + margin)

        return gray[y1:y2, x1:x2], mask[y1:y2, x1:x2], (x1, y1)

    def _compute_auto_scale(self, obj_h: int, obj_w: int, target_h: int, target_w: int, rng: np.random.Generator) -> float:
        """计算自动缩放比例，使物体在目标图像中占合适大小。

        根据 target_occupancy 参数计算缩放比例：
        - 物体应该占目标图像的 target_occupancy[0] ~ target_occupancy[1] 比例
        - 然后在此基础上乘以 scale_range 的随机因子
        """
        # 计算物体占目标图像的比例
        target_size = min(target_h, target_w)
        obj_size = max(obj_h, obj_w)

        if obj_size == 0:
            return 1.0

        # 目标物体应该占目标图像的比例
        desired_occupancy = rng.uniform(*self.target_occupancy)
        desired_size = target_size * desired_occupancy

        # 基础缩放比例
        base_scale = desired_size / obj_size

        # 额外随机缩放
        extra_scale = rng.uniform(*self.scale_range)

        return base_scale * extra_scale

    def _random_scale(self, gray: np.ndarray, mask: np.ndarray, rng: np.random.Generator,
                      target_h: int = 224, target_w: int = 224) -> Tuple[np.ndarray, np.ndarray, float]:
        """随机缩放目标物体，自动计算合适的缩放比例。

        Returns:
            (gray_scaled, mask_scaled, scale_factor)
        """
        obj_h, obj_w = gray.shape
        scale = self._compute_auto_scale(obj_h, obj_w, target_h, target_w, rng)

        new_h = max(1, int(obj_h * scale))
        new_w = max(1, int(obj_w * scale))

        gray_scaled = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask_scaled = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        return gray_scaled, mask_scaled, scale

    def _blur_mask_edges(self, mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """对 mask 边缘做高斯模糊，实现柔化效果。"""
        sigma = rng.uniform(*self.edge_blur_range)
        ksize = int(sigma * 6) | 1
        ksize = max(ksize, 3)
        mask_f = mask.astype(np.float32)
        mask_blurred = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)
        return np.clip(mask_blurred, 0, 255).astype(np.uint8)

    def generate(self, image_size: Tuple[int, int], rng: np.random.Generator) -> TargetRender:
        h, w = image_size

        # 1. 随机选一张图片
        idx = rng.integers(0, len(self._image_paths))
        path = self._image_paths[idx]

        # 2. 加载图片和 mask
        gray, mask = self._load_image_and_mask(path)

        # 3. 裁剪到内容区域
        gray, mask, (ox, oy) = self._crop_to_content(gray, mask)

        # 4. 自动缩放（根据目标图像尺寸）
        gray, mask, scale = self._random_scale(gray, mask, rng, target_h=h, target_w=w)

        # 5. 柔化 mask 边缘
        mask_soft = self._blur_mask_edges(mask, rng)

        # 6. 创建全尺寸的输出数组
        out_gray = np.zeros((h, w), dtype=np.float32)
        out_mask = np.zeros((h, w), dtype=np.uint8)

        # 将目标放在图像中心
        obj_h, obj_w = gray.shape
        paste_x = (w - obj_w) // 2
        paste_y = (h - obj_h) // 2

        # 处理目标超出图像边界的情况
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(obj_w, w - paste_x)
        src_y2 = min(obj_h, h - paste_y)

        dst_x1 = paste_x + src_x1
        dst_y1 = paste_y + src_y1
        dst_x2 = paste_x + src_x2
        dst_y2 = paste_y + src_y2

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            out_gray[dst_y1:dst_y2, dst_x1:dst_x2] = gray[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
            out_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask_soft[src_y1:src_y2, src_x1:src_x2]

        # 参考点 = 图像中心
        reference_point = (w / 2.0, h / 2.0)

        return TargetRender(
            mask=out_mask,
            reference_point=reference_point,
            label=self.label,
            texture=out_gray,
        )


class SpermTailFromImageGenerator(RealImageTargetGenerator):
    """从整精子图片中提取精子尾部，参考点为尾部尖端。

    关键流程：
    1. 加载预计算的 tail_tip 位置（从 JSON 文件）
    2. 裁剪到内容区域，同时更新 tail_tip 坐标
    3. 自动缩放到目标图像大小，同时缩放 tail_tip 坐标
    4. 粘贴到目标图像中，同时更新 tail_tip 坐标

    预计算文件由 scripts/preprocess_sperm_tail_tips.py 生成。
    """

    def __init__(
        self,
        image_dir: str | Path,
        tail_tips_file: str | Path | None = None,
        scale_range: Tuple[float, float] = (0.3, 0.6),
        target_occupancy: Tuple[float, float] = (0.4, 0.7),
        crop_margin: int = 5,
        edge_blur_range: Tuple[float, float] = (0.8, 2.5),
    ):
        """
        Args:
            image_dir: 整精子图片目录。
            tail_tips_file: 预计算的 tail_tip JSON 文件路径。
                如果为 None，自动查找 data/sperm_tail_tips.json。
            scale_range: 额外随机缩放范围。
            target_occupancy: 目标物体占图像的比例范围。
            crop_margin: 裁剪边距。
            edge_blur_range: 边缘模糊范围。
        """
        super().__init__(
            image_dir=image_dir,
            scale_range=scale_range,
            target_occupancy=target_occupancy,
            label="sperm_tail",
            crop_margin=crop_margin,
            edge_blur_range=edge_blur_range,
        )

        # 加载预计算的 tail_tip 位置
        self._tail_tips: dict[str, tuple] = {}
        self._load_tail_tips(tail_tips_file)

    def _load_tail_tips(self, tail_tips_file: str | Path | None):
        """加载预计算的 tail_tip 位置。"""
        import json as json_module

        if tail_tips_file is None:
            # 自动查找
            candidates = [
                Path("data/sperm_tail_tips.json"),
                Path("data/pre-individual-obj/individual_obj/sperm_tail_tips.json"),
            ]
            for candidate in candidates:
                if candidate.exists():
                    tail_tips_file = candidate
                    break

        if tail_tips_file is None:
            print("警告：未找到预计算的 tail_tip 文件，将使用在线计算")
            return

        tail_tips_file = Path(tail_tips_file)
        if not tail_tips_file.exists():
            print(f"警告：tail_tip 文件不存在: {tail_tips_file}，将使用在线计算")
            return

        with open(tail_tips_file, "r", encoding="utf-8") as f:
            data = json_module.load(f)

        for item in data:
            if "error" not in item and "tail_tip" in item:
                self._tail_tips[item["file"]] = tuple(item["tail_tip"])

        print(f"加载了 {len(self._tail_tips)} 个预计算的 tail_tip 位置")

    def _find_tail_tip_online(self, mask: np.ndarray) -> Tuple[float, float]:
        """在线计算 tail_tip 位置（当预计算文件不可用时的回退方案）。

        骨架化找端点，离质心最远的是尾部尖端。
        """
        skeleton = self._skeletonize(mask)

        h, w = skeleton.shape
        endpoints = []
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if skeleton[y, x] == 0:
                    continue
                neighbors = np.sum(skeleton[y - 1:y + 2, x - 1:x + 2] > 0) - 1
                if neighbors == 1:
                    endpoints.append((x, y))

        if not endpoints:
            M = cv2.moments(mask)
            if M["m00"] > 0:
                return (M["m10"] / M["m00"], M["m01"] / M["m00"])
            return (mask.shape[1] / 2, mask.shape[0] / 2)

        M = cv2.moments(mask)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cx = w / 2
            cy = h / 2

        tail_tip = max(endpoints, key=lambda p: (p[0] - cx) ** 2 + (p[1] - cy) ** 2)
        return (float(tail_tip[0]), float(tail_tip[1]))

    @staticmethod
    def _skeletonize(img: np.ndarray) -> np.ndarray:
        """使用形态学操作实现骨架化。"""
        img = img.copy()
        skel = np.zeros(img.shape, np.uint8)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, element)
            temp = cv2.dilate(eroded, element)
            temp = cv2.subtract(img, temp)
            skel = cv2.bitwise_or(skel, temp)
            img = eroded.copy()
            if cv2.countNonZero(img) == 0:
                break
        return skel

    def _get_tail_tip(self, filename: str, mask: np.ndarray) -> Tuple[float, float]:
        """获取 tail_tip 位置，优先使用预计算值。"""
        if filename in self._tail_tips:
            return self._tail_tips[filename]
        return self._find_tail_tip_online(mask)

    def _transform_point(self, point: Tuple[float, float],
                         crop_offset: Tuple[int, int],
                         scale: float,
                         paste_offset: Tuple[int, int]) -> Tuple[float, float]:
        """将点坐标从原图空间变换到最终输出空间。

        变换链：原图 -> 裁剪 -> 缩放 -> 粘贴
        """
        x = point[0] - crop_offset[0]
        y = point[1] - crop_offset[1]
        x *= scale
        y *= scale
        x += paste_offset[0]
        y += paste_offset[1]
        return (x, y)

    @staticmethod
    def _clamp_to_mask(point: Tuple[float, float], mask: np.ndarray) -> Tuple[float, float]:
        """确保点在 mask 内部。如果不在，找最近的 mask 像素。"""
        h, w = mask.shape
        px, py = int(round(point[0])), int(round(point[1]))

        # 如果已经在 mask 上，直接返回
        if 0 <= px < w and 0 <= py < h and mask[py, px] > 0:
            return point

        # clamp 到图像范围
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)

        # 如果 clamp 后在 mask 上，返回 clamp 后的值
        if mask[py, px] > 0:
            return (float(px), float(py))

        # 找最近的 mask 像素
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return (float(w / 2), float(h / 2))

        dists = (xs - point[0]) ** 2 + (ys - point[1]) ** 2
        nearest_idx = np.argmin(dists)
        return (float(xs[nearest_idx]), float(ys[nearest_idx]))

    def generate(self, image_size: Tuple[int, int], rng: np.random.Generator) -> TargetRender:
        """生成精子尾部目标，参考点为尾部尖端。

        流程：
        1. 加载原图
        2. 获取 tail_tip 位置（预计算或在线计算）
        3. 裁剪到内容区域
        4. 自动缩放
        5. 粘贴到目标图像
        6. 变换 tail_tip 坐标到最终位置
        """
        h, w = image_size

        # 1. 随机选一张图片
        idx = rng.integers(0, len(self._image_paths))
        path = self._image_paths[idx]

        # 2. 加载图片和 mask（原图空间）
        gray, mask = self._load_image_and_mask(path)

        # 3. 获取 tail_tip 位置（原图坐标）
        tail_tip_original = self._get_tail_tip(path.name, mask)

        # 4. 裁剪到内容区域
        gray, mask, (ox, oy) = self._crop_to_content(gray, mask)

        # 5. 自动缩放
        gray, mask, scale = self._random_scale(gray, mask, rng, target_h=h, target_w=w)

        # 6. 柔化 mask 边缘
        mask_soft = self._blur_mask_edges(mask, rng)

        # 7. 创建全尺寸的输出数组
        out_gray = np.zeros((h, w), dtype=np.float32)
        out_mask = np.zeros((h, w), dtype=np.uint8)

        # 将目标放在图像中心
        obj_h, obj_w = gray.shape
        paste_x = (w - obj_w) // 2
        paste_y = (h - obj_h) // 2

        # 处理目标超出图像边界的情况
        src_x1 = max(0, -paste_x)
        src_y1 = max(0, -paste_y)
        src_x2 = min(obj_w, w - paste_x)
        src_y2 = min(obj_h, h - paste_y)

        dst_x1 = paste_x + src_x1
        dst_y1 = paste_y + src_y1
        dst_x2 = paste_x + src_x2
        dst_y2 = paste_y + src_y2

        if dst_x2 > dst_x1 and dst_y2 > dst_y1:
            out_gray[dst_y1:dst_y2, dst_x1:dst_x2] = gray[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
            out_mask[dst_y1:dst_y2, dst_x1:dst_x2] = mask_soft[src_y1:src_y2, src_x1:src_x2]

        # 8. 变换 tail_tip 到最终输出坐标
        tail_tip_final = self._transform_point(
            tail_tip_original,
            crop_offset=(ox, oy),
            scale=scale,
            paste_offset=(paste_x, paste_y),
        )

        # 9. 修正 tail_tip：确保在 mask 内部
        #    缩放可能导致 tip 超出 mask 边界，需要 clamp 并找最近的 mask 像素
        tail_tip_final = self._clamp_to_mask(tail_tip_final, out_mask)

        return TargetRender(
            mask=out_mask,
            reference_point=tail_tip_final,
            label="sperm_tail",
            texture=out_gray,
        )


def cv2_circle_fill(img: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    """Fill a circle on a numpy array without cv2 dependency for pure array ops."""
    h, w = img.shape
    cx, cy = center
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    img[dist <= radius] = 255


# Registries for generator lookup
TASK_TO_GENERATOR = {
    "microsphere": MicrosphereGenerator,
}

GENERATOR_REGISTRY = {
    "MicrosphereGenerator": MicrosphereGenerator,
    "RealImageTargetGenerator": RealImageTargetGenerator,
    "SpermTailFromImageGenerator": SpermTailFromImageGenerator,
}


def get_target_generator_for_task(task_name: str) -> TargetGenerator:
    """Create a default target generator for a named task.

    Args:
        task_name: Lowercase task name, e.g. "microsphere".

    Returns:
        TargetGenerator instance with default parameters.

    Raises:
        ValueError: If the task name is not recognized.
    """
    cls = TASK_TO_GENERATOR.get(task_name)
    if cls is None:
        raise ValueError(
            f"Unknown task '{task_name}'. Available: {list(TASK_TO_GENERATOR.keys())}"
        )
    return cls()


def create_target_generator(generator_name: str, params: dict) -> TargetGenerator:
    """Create a target generator from a class name and parameter dict.

    Args:
        generator_name: Class name, e.g. "MicrosphereGenerator".
        params: Keyword arguments for the generator constructor.

    Returns:
        TargetGenerator instance.

    Raises:
        ValueError: If the generator name is not recognized.
    """
    cls = GENERATOR_REGISTRY.get(generator_name)
    if cls is None:
        raise ValueError(
            f"Unknown generator '{generator_name}'. Available: {list(GENERATOR_REGISTRY.keys())}"
        )
    return cls(**params)
