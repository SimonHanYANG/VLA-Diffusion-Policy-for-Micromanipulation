from typing import Optional, Tuple

import cv2
import numpy as np

from src.simulator.background import BackgroundGenerator
from src.simulator.noise import NoiseApplicator
from src.simulator.optics import OpticsProcessor
from src.utils.config import DomainRandConfig, NoiseConfig, OpticsConfig


class MicroscopeRenderer:
    """Composes the final microscope image.

    Composition order:
    1. Generate background
    2. Paste target object with edge blending (soft alpha, brightness adaptation, edge fade)
    3. Draw red laser dot at image center
    4. Apply optical effects (vignetting, PSF, chromatic aberration)
    5. Apply domain randomization (brightness, contrast, blur)
    6. Apply noise models
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        background: BackgroundGenerator,
        noise: NoiseApplicator | None = None,
        domain_rand: DomainRandConfig = DomainRandConfig(),
        optics: OpticsConfig | None = None,
        # 边缘融合参数
        blend_factor_range: Tuple[float, float] = (0.6, 0.8),
        edge_fade_ratio: float = 0.15,
        brightness_adapt_range: Tuple[float, float] = (0.5, 2.0),
        edge_blur_kernel_range: Tuple[int, int] = (3, 9),
        edge_blur_sigma_range: Tuple[float, float] = (0.8, 2.5),
    ):
        self.image_size = image_size
        self.background = background
        self.noise = noise  # 可以为 None，表示不加噪声
        self.domain_rand = domain_rand
        self.optics = OpticsProcessor(optics) if optics is not None else None
        self.laser_position = (image_size[1] // 2, image_size[0] // 2)
        self.laser_radius = max(1, int(3 * min(image_size) / 224))
        self.laser_color_bgr = (230, 230, 230)  # Bright gray dot

        # 边缘融合参数
        self.blend_factor_range = blend_factor_range
        self.edge_fade_ratio = edge_fade_ratio
        self.brightness_adapt_range = brightness_adapt_range
        self.edge_blur_kernel_range = edge_blur_kernel_range
        self.edge_blur_sigma_range = edge_blur_sigma_range

    def set_laser_position(self, x: int, y: int) -> None:
        self.laser_position = (x, y)

    def render(
        self,
        target_mask: np.ndarray,
        target_position: Tuple[float, float],
        rng: np.random.Generator,
        target_texture: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Render a complete microscope image with realistic edge blending.

        Args:
            target_mask: (H, W) uint8 mask of the target (255=target).
            target_position: (x, y) center position of the target in the rendered image.
            rng: random generator for noise and domain randomization.
            target_texture: (H, W) float32 intensity map. If None, uses default (200).

        Returns:
            (H, W, 3) BGR uint8 image (grayscale × 3 channels).
        """
        h, w = self.image_size

        # 1. Background (grayscale)
        bg = self.background.generate((w, h), rng)

        # 2. Paste target with edge blending
        gray = self._blend_target_onto_background(
            bg, target_mask, target_position, target_texture, rng
        )

        # 3. Convert grayscale to 3-channel (grayscale × 3)
        image = np.stack([gray, gray, gray], axis=-1)  # (H, W, 3)

        # 4. Red laser dot at center (on 3-channel image)
        cv2.circle(image, self.laser_position, self.laser_radius, self.laser_color_bgr, -1)

        # 5. Optical effects (vignetting, PSF, chromatic aberration)
        if self.optics is not None:
            image = self.optics.apply(image, rng)

        # 6. Domain randomization
        if self.domain_rand is not None:
            image = self._apply_domain_randomization(image, rng)

        # 7. Noise
        if self.noise is not None:
            image = self.noise.apply(image, rng)

        return image

    def _blend_target_onto_background(
        self,
        bg: np.ndarray,
        target_mask: np.ndarray,
        target_position: Tuple[float, float],
        target_texture: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> np.ndarray:
        """将目标物体融合到背景上，使用多层融合策略消除贴图感。

        融合策略：
        1. 柔化 Alpha 边缘（高斯模糊 mask）
        2. 局部亮度自适应（匹配背景亮度）
        3. 边缘渐晕融合（边缘渐变透明）
        4. 纹理叠加（非直接替换）
        """
        h, w = self.image_size
        target_h, target_w = target_mask.shape

        # 计算粘贴位置
        tx = int(round(target_position[0] - target_w / 2))
        ty = int(round(target_position[1] - target_h / 2))

        # 计算有效区域
        src_x1 = max(0, -tx)
        src_y1 = max(0, -ty)
        src_x2 = min(target_w, w - tx)
        src_y2 = min(target_h, h - ty)
        dst_x1 = max(0, tx)
        dst_y1 = max(0, ty)
        dst_x2 = min(w, tx + target_w)
        dst_y2 = min(h, ty + target_h)

        gray = bg.copy().astype(np.float32)

        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return gray.astype(np.uint8)

        # 提取目标区域
        src_mask = target_mask[src_y1:src_y2, src_x1:src_x2]
        dst_region = gray[dst_y1:dst_y2, dst_x1:dst_x2]

        # 使用纹理或默认值
        if target_texture is not None:
            tex_region = target_texture[src_y1:src_y2, src_x1:src_x2]
        else:
            tex_region = np.full_like(dst_region, 200.0, dtype=np.float32)

        # === 步骤 1: 柔化 Alpha 边缘 ===
        mask_f = src_mask.astype(np.float32)
        ksize = rng.integers(self.edge_blur_kernel_range[0], self.edge_blur_kernel_range[1] + 1)
        if ksize % 2 == 0:
            ksize += 1
        sigma = rng.uniform(*self.edge_blur_sigma_range)
        alpha = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma) / 255.0

        # === 步骤 2: 局部亮度自适应 ===
        # 提取目标区域对应背景的局部亮度统计
        mask_binary = src_mask > 128
        if np.any(mask_binary):
            bg_mean = np.mean(dst_region[mask_binary])
            obj_mean = np.mean(tex_region[mask_binary])

            if obj_mean > 5:  # 避免除零
                brightness_ratio = bg_mean / obj_mean
                brightness_ratio = np.clip(
                    brightness_ratio,
                    self.brightness_adapt_range[0],
                    self.brightness_adapt_range[1],
                )
                # 调整目标亮度使其与背景匹配，但保持目标略暗
                adjusted_texture = tex_region * brightness_ratio * 0.6
                adjusted_texture = np.clip(adjusted_texture, 0, 255)
            else:
                adjusted_texture = tex_region
        else:
            adjusted_texture = tex_region

        # === 步骤 3: 边缘渐晕融合 ===
        # 在目标边缘区域额外添加一层渐变，模拟光学模糊
        if np.any(mask_binary):
            # 计算距离变换（到边缘的距离）
            dist = cv2.distanceTransform(src_mask, cv2.DIST_L2, 5)
            max_dist = dist.max()
            if max_dist > 0:
                # 最外 edge_fade_ratio 比例的区域渐变
                fade_threshold = max_dist * self.edge_fade_ratio
                edge_fade = np.clip(dist / (fade_threshold + 1e-6), 0, 1)
            else:
                edge_fade = np.ones_like(dist, dtype=np.float32)
        else:
            edge_fade = np.ones_like(alpha, dtype=np.float32)

        # 最终 alpha = 柔化 alpha × 边缘渐晕
        final_alpha = alpha * edge_fade

        # === 步骤 4: 纹理叠加（非直接替换） ===
        # 让目标"浮"在背景上，blend_factor 控制目标与背景的混合程度
        blend_factor = rng.uniform(*self.blend_factor_range)
        blended_value = blend_factor * adjusted_texture + (1 - blend_factor) * dst_region

        # 最终合成
        dst_region[:] = (final_alpha * blended_value + (1 - final_alpha) * dst_region)

        return np.clip(gray, 0, 255).astype(np.uint8)

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
