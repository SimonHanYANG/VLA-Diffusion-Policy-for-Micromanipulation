"""测试新的真实图片融合仿真功能。

测试内容：
1. MultiImageBackground 能否正确加载和生成背景
2. RealImageTargetGenerator 能否正确加载真实图片
3. SpermTailFromImageGenerator 能否正确提取尾部尖端
4. MicroscopeRenderer 的边缘融合逻辑是否正常工作
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator.background import MultiImageBackground
from src.simulator.renderer import MicroscopeRenderer
from src.simulator.targets import (
    MicrosphereGenerator,
    RealImageTargetGenerator,
    SpermTailFromImageGenerator,
)
from src.utils.config import DomainRandConfig, OpticsConfig


class TestMultiImageBackground:
    """测试 MultiImageBackground 类。"""

    def test_generate_returns_correct_shape(self):
        """测试生成的背景尺寸是否正确。"""
        bg = MultiImageBackground(
            image_paths=[
                "data/backgrounds_test/bg_00001.png",
                "data/backgrounds_test/bg_00002.png",
                "data/backgrounds_test/bg_00003.png",
            ]
        )
        rng = np.random.default_rng(42)
        result = bg.generate((224, 224), rng)

        assert result.shape == (224, 224)
        assert result.dtype == np.uint8

    def test_generate_returns_grayscale(self):
        """测试生成的背景是否为灰度图。"""
        bg = MultiImageBackground(
            image_paths=[
                "data/backgrounds_test/bg_00001.png",
                "data/backgrounds_test/bg_00002.png",
                "data/backgrounds_test/bg_00003.png",
            ]
        )
        rng = np.random.default_rng(42)
        result = bg.generate((224, 224), rng)

        # 灰度图应该是单通道
        assert len(result.shape) == 2

    def test_different_seeds_give_different_results(self):
        """测试不同种子是否产生不同的结果。"""
        bg = MultiImageBackground(
            image_paths=[
                "data/backgrounds_test/bg_00001.png",
                "data/backgrounds_test/bg_00002.png",
                "data/backgrounds_test/bg_00003.png",
            ]
        )

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        result1 = bg.generate((224, 224), rng1)
        result2 = bg.generate((224, 224), rng2)

        # 不同种子应该产生不同的结果（概率极高）
        assert not np.array_equal(result1, result2)


class TestRealImageTargetGenerator:
    """测试 RealImageTargetGenerator 类。"""

    def test_embryo_generator_loads_images(self):
        """测试胚胎图片加载是否正常。"""
        embryo_dir = Path("data/pre-individual-obj/individual_obj/embryo")
        if not embryo_dir.exists():
            pytest.skip("胚胎图片目录不存在")

        gen = RealImageTargetGenerator(
            image_dir=embryo_dir,
            scale_range=(0.8, 1.2),
            label="embryo",
        )

        assert len(gen.image_paths) > 0

    def test_generate_returns_correct_structure(self):
        """测试生成的目标结构是否正确。"""
        embryo_dir = Path("data/pre-individual-obj/individual_obj/embryo")
        if not embryo_dir.exists():
            pytest.skip("胚胎图片目录不存在")

        gen = RealImageTargetGenerator(
            image_dir=embryo_dir,
            scale_range=(0.8, 1.2),
            label="embryo",
        )

        rng = np.random.default_rng(42)
        target = gen.generate((224, 224), rng)

        assert target.mask.shape == (224, 224)
        assert target.mask.dtype == np.uint8
        assert target.texture.shape == (224, 224)
        assert target.texture.dtype == np.float32
        assert target.label == "embryo"
        assert isinstance(target.reference_point, tuple)
        assert len(target.reference_point) == 2


class TestSpermTailFromImageGenerator:
    """测试 SpermTailFromImageGenerator 类。"""

    def test_sperm_tail_generator_loads_images(self):
        """测试整精子图片加载是否正常。"""
        whole_sperm_dir = Path("data/pre-individual-obj/individual_obj/whole_sperm")
        if not whole_sperm_dir.exists():
            pytest.skip("整精子图片目录不存在")

        gen = SpermTailFromImageGenerator(
            image_dir=whole_sperm_dir,
            scale_range=(0.8, 1.2),
        )

        assert len(gen.image_paths) > 0

    def test_generate_returns_correct_structure(self):
        """测试生成的精子尾部结构是否正确。"""
        whole_sperm_dir = Path("data/pre-individual-obj/individual_obj/whole_sperm")
        if not whole_sperm_dir.exists():
            pytest.skip("整精子图片目录不存在")

        gen = SpermTailFromImageGenerator(
            image_dir=whole_sperm_dir,
            scale_range=(0.8, 1.2),
        )

        rng = np.random.default_rng(42)
        target = gen.generate((224, 224), rng)

        assert target.mask.shape == (224, 224)
        assert target.mask.dtype == np.uint8
        assert target.texture.shape == (224, 224)
        assert target.texture.dtype == np.float32
        assert target.label == "sperm_tail"
        assert isinstance(target.reference_point, tuple)
        assert len(target.reference_point) == 2


class TestMicroscopeRendererBlending:
    """测试 MicroscopeRenderer 的边缘融合功能。"""

    def test_render_with_realistic_blending(self):
        """测试渲染器是否能正常工作。"""
        bg = MultiImageBackground(
            image_paths=[
                "data/backgrounds_test/bg_00001.png",
                "data/backgrounds_test/bg_00002.png",
                "data/backgrounds_test/bg_00003.png",
            ]
        )

        renderer = MicroscopeRenderer(
            image_size=(224, 224),
            background=bg,
            noise=None,
            domain_rand=DomainRandConfig(brightness_range=(1.0, 1.0), contrast_range=(1.0, 1.0)),
            optics=None,
            blend_factor_range=(0.6, 0.8),
            edge_fade_ratio=0.15,
        )

        # 创建简单的测试目标
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        texture = np.full((50, 50), 100.0, dtype=np.float32)

        rng = np.random.default_rng(42)
        image = renderer.render(
            target_mask=mask,
            target_position=(112, 112),
            rng=rng,
            target_texture=texture,
        )

        assert image.shape == (224, 224, 3)
        assert image.dtype == np.uint8

    def test_render_preserves_grayscale(self):
        """测试渲染后的图像是否保持灰度（3通道相同值）。"""
        bg = MultiImageBackground(
            image_paths=[
                "data/backgrounds_test/bg_00001.png",
            ]
        )

        renderer = MicroscopeRenderer(
            image_size=(224, 224),
            background=bg,
            noise=None,
            domain_rand=DomainRandConfig(brightness_range=(1.0, 1.0), contrast_range=(1.0, 1.0)),
            optics=None,
        )

        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        texture = np.full((50, 50), 100.0, dtype=np.float32)

        rng = np.random.default_rng(42)
        image = renderer.render(
            target_mask=mask,
            target_position=(112, 112),
            rng=rng,
            target_texture=texture,
        )

        # 灰度图 × 3 通道：3个通道应该相同
        # 注意：由于激光点是红色的，中心区域会有差异
        # 我们检查大部分区域（非激光点区域）
        center_x, center_y = 112, 112
        for x in range(10, 214, 20):
            for y in range(10, 214, 20):
                if abs(x - center_x) > 10 or abs(y - center_y) > 10:
                    b, g, r = image[y, x]
                    # 在非激光点区域，3个通道应该相同
                    # 允许小的数值误差（由于浮点运算）
                    assert abs(int(b) - int(g)) <= 1
                    assert abs(int(g) - int(r)) <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
