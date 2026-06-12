"""Tests for simulator components: stage, targets, noise, backgrounds, environment."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from src.simulator.stage import StageSimulator
from src.simulator.targets import (
    MicrosphereGenerator,
    RealImageTargetGenerator,
    SpermTailFromImageGenerator,
    create_target_generator,
    get_target_generator_for_task,
)
from src.simulator.noise import NoiseApplicator, NoiseConfig
from src.simulator.background import PerlinNoiseBackground, SingleImageBackground
from src.simulator.environment import MicroscopeEnvironment
from src.utils.config import SimulatorConfig


# ------------------------------------------------------------------
# StageSimulator
# ------------------------------------------------------------------

class TestStageSimulator:
    def test_initial_position_is_zero(self):
        stage = StageSimulator()
        stage.connect()
        x, y = stage.get_xy_position()
        assert x == 0.0
        assert y == 0.0

    def test_move_absolute(self):
        stage = StageSimulator()
        stage.connect()
        stage.move_xy_to_absolute(100.0, 200.0)
        x, y = stage.get_xy_position()
        assert x == 100.0
        assert y == 200.0

    def test_move_relative(self):
        stage = StageSimulator()
        stage.connect()
        stage.move_xy_relative(50.0, -30.0)
        x, y = stage.get_xy_position()
        assert x == 50.0
        assert y == -30.0

    def test_clamp_to_range(self):
        stage = StageSimulator(x_range=(-100.0, 100.0), y_range=(-50.0, 50.0))
        stage.connect()
        stage.move_xy_to_absolute(200.0, -200.0)
        x, y = stage.get_xy_position()
        assert x == 100.0
        assert y == -50.0

    def test_reset_position(self):
        stage = StageSimulator()
        stage.connect()
        stage.move_xy_to_absolute(500.0, 500.0)
        stage.reset_position()
        x, y = stage.get_xy_position()
        assert x == 0.0
        assert y == 0.0

    def test_connect_dispose_warning(self):
        stage = StageSimulator()
        with pytest.warns(UserWarning, match="not connected"):
            stage.move_xy_to_absolute(10.0, 10.0)


# ------------------------------------------------------------------
# Target Generators
# ------------------------------------------------------------------

class TestTargetGenerators:
    IMAGE_SIZE = (224, 224)
    RNG = np.random.default_rng(42)

    def test_microsphere_generator(self):
        gen = MicrosphereGenerator()
        target = gen.generate(self.IMAGE_SIZE, self.RNG)
        assert target.mask.shape == self.IMAGE_SIZE
        assert target.mask.dtype == np.uint8
        assert target.label == "microsphere"
        assert len(target.reference_point) == 2
        assert np.any(target.mask > 0)  # non-empty

    def test_create_target_generator_with_params(self):
        gen = create_target_generator("MicrosphereGenerator", {"radius_min": 10.0, "radius_max": 20.0})
        target = gen.generate(self.IMAGE_SIZE, self.RNG)
        assert target.label == "microsphere"

    def test_create_target_generator_unknown(self):
        with pytest.raises(ValueError, match="Unknown generator"):
            create_target_generator("NonExistentGenerator", {})

    def test_get_target_generator_for_task_microsphere(self):
        gen = get_target_generator_for_task("microsphere")
        target = gen.generate(self.IMAGE_SIZE, self.RNG)
        assert target.label == "microsphere"

    def test_get_target_generator_for_task_unknown(self):
        with pytest.raises(ValueError, match="Unknown task"):
            get_target_generator_for_task("invalid_task")


# ------------------------------------------------------------------
# Real Image Target Generators
# ------------------------------------------------------------------

class TestRealImageTargetGenerators:
    IMAGE_SIZE = (224, 224)
    RNG = np.random.default_rng(42)

    @pytest.fixture
    def embryo_dir(self):
        d = Path("data/pre-individual-obj/individual_obj/embryo")
        if not d.exists():
            pytest.skip("Real embryo images not found")
        return d

    @pytest.fixture
    def sperm_head_dir(self):
        d = Path("data/pre-individual-obj/individual_obj/sperm_head")
        if not d.exists():
            pytest.skip("Real sperm head images not found")
        return d

    @pytest.fixture
    def whole_sperm_dir(self):
        d = Path("data/pre-individual-obj/individual_obj/whole_sperm")
        if not d.exists():
            pytest.skip("Real whole sperm images not found")
        return d

    def test_real_image_embryo(self, embryo_dir):
        gen = RealImageTargetGenerator(image_dir=embryo_dir, label="embryo")
        target = gen.generate(self.IMAGE_SIZE, self.RNG)
        assert target.mask.shape == self.IMAGE_SIZE
        assert target.label == "embryo"
        assert np.any(target.mask > 0)

    def test_real_image_sperm_head(self, sperm_head_dir):
        gen = RealImageTargetGenerator(image_dir=sperm_head_dir, label="sperm_head")
        target = gen.generate(self.IMAGE_SIZE, self.RNG)
        assert target.mask.shape == self.IMAGE_SIZE
        assert target.label == "sperm_head"
        assert np.any(target.mask > 0)

    def test_sperm_tail_from_image(self, whole_sperm_dir):
        gen = SpermTailFromImageGenerator(image_dir=whole_sperm_dir)
        target = gen.generate(self.IMAGE_SIZE, self.RNG)
        assert target.mask.shape == self.IMAGE_SIZE
        assert target.label == "sperm_tail"
        assert np.any(target.mask > 0)

    def test_real_image_in_environment(self, embryo_dir):
        gen = RealImageTargetGenerator(image_dir=embryo_dir, label="embryo")
        config = SimulatorConfig(image_size=(224, 224))
        env = MicroscopeEnvironment(config=config, target_generator=gen)
        obs, info = env.reset(seed=42)
        assert obs.shape == (224, 224, 3)
        assert info["target_label"] == "embryo"


# ------------------------------------------------------------------
# NoiseApplicator
# ------------------------------------------------------------------

class TestNoiseApplicator:
    def test_apply_gaussian(self):
        config = NoiseConfig(gaussian_std=5.0, salt_pepper_prob=0.0, flicker_prob=0.0, motion_blur_prob=0.0)
        applicator = NoiseApplicator(config)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        rng = np.random.default_rng(42)
        noisy = applicator.apply(img, rng)
        assert noisy.shape == img.shape
        assert noisy.dtype == np.uint8
        # With Gaussian noise, statistics should differ slightly
        assert not np.array_equal(noisy, img)

    def test_apply_no_noise(self):
        config = NoiseConfig(gaussian_std=0.0, salt_pepper_prob=0.0, flicker_prob=0.0, motion_blur_prob=0.0)
        applicator = NoiseApplicator(config)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        rng = np.random.default_rng(42)
        noisy = applicator.apply(img, rng)
        assert np.array_equal(noisy, img)

    def test_output_dtype_uint8(self):
        applicator = NoiseApplicator()
        img = np.ones((50, 50, 3), dtype=np.uint8) * 100
        rng = np.random.default_rng(42)
        noisy = applicator.apply(img, rng)
        assert noisy.dtype == np.uint8

    def test_salt_pepper_range(self):
        config = NoiseConfig(gaussian_std=0.0, salt_pepper_prob=1.0, flicker_prob=0.0,
                             motion_blur_prob=0.0, fixed_pattern_prob=0.0)
        applicator = NoiseApplicator(config)
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        # The salt-pepper gate triggers ~50% of the time; try multiple seeds
        found_active = False
        for seed in range(100):
            rng = np.random.default_rng(seed)
            noisy = applicator.apply(img.copy(), rng)
            if not np.array_equal(noisy, img):
                found_active = True
                # With salt_pepper_prob=1.0 and gate active, all pixels become 0 or 255
                assert np.all((noisy == 0) | (noisy == 255))
                break
        assert found_active, "Salt-pepper gate never triggered in 100 seeds"


# ------------------------------------------------------------------
# Background
# ------------------------------------------------------------------

class TestPerlinNoiseBackground:
    def test_generate_shape(self):
        bg = PerlinNoiseBackground()
        rng = np.random.default_rng(42)
        # size is (width, height), output is (height, width)
        result = bg.generate((256, 128), rng)
        assert result.shape == (128, 256)
        assert result.dtype == np.uint8

    def test_generate_square(self):
        bg = PerlinNoiseBackground()
        rng = np.random.default_rng(42)
        result = bg.generate((224, 224), rng)
        assert result.shape == (224, 224)

    def test_value_range(self):
        bg = PerlinNoiseBackground(intensity_min=80, intensity_max=160)
        rng = np.random.default_rng(42)
        result = bg.generate((100, 100), rng)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_deterministic_with_seed(self):
        bg = PerlinNoiseBackground()
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)
        res1 = bg.generate((64, 64), rng1)
        res2 = bg.generate((64, 64), rng2)
        assert np.array_equal(res1, res2)


# ------------------------------------------------------------------
# MicroscopeEnvironment
# ------------------------------------------------------------------

class TestMicroscopeEnvironment:
    def test_reset_returns_image_and_info(self):
        config = SimulatorConfig(image_size=(224, 224))
        gen = get_target_generator_for_task("microsphere")
        env = MicroscopeEnvironment(config=config, target_generator=gen)
        obs, info = env.reset(seed=42)
        assert obs.shape == (224, 224, 3)
        assert obs.dtype == np.uint8
        assert "target_position" in info
        assert "stage_position" in info
        assert "step" in info
        assert "distance" in info

    def test_step_changes_observation(self):
        config = SimulatorConfig(image_size=(224, 224))
        gen = get_target_generator_for_task("microsphere")
        env = MicroscopeEnvironment(config=config, target_generator=gen)
        obs1, info1 = env.reset(seed=42)
        action = np.array([10.0, 5.0], dtype=np.float64)
        obs2, reward, done, info2 = env.step(action)
        assert obs2.shape == obs1.shape
        assert info2["step"] == 1
        assert isinstance(reward, float)

    def test_done_on_close_approach(self):
        config = SimulatorConfig(image_size=(224, 224), max_steps_per_episode=200)
        gen = get_target_generator_for_task("microsphere")
        env = MicroscopeEnvironment(config=config, target_generator=gen)
        obs, info = env.reset(seed=42)
        # Move directly toward target
        target = info["target_position"]
        laser = np.array([112.0, 112.0])
        delta = target - laser
        action = delta * 0.5
        obs, reward, done, info = env.step(action)
        # With 0.5 of the distance, should not be done yet (target is 50-200px away)
        assert not done or info.get("distance", 999) < config.success_tolerance_px

    def test_done_on_max_steps(self):
        config = SimulatorConfig(image_size=(224, 224), max_steps_per_episode=5)
        gen = get_target_generator_for_task("microsphere")
        env = MicroscopeEnvironment(config=config, target_generator=gen)
        env.reset(seed=42)
        for _ in range(5):
            action = np.array([1.0, 1.0], dtype=np.float64)
            obs, reward, done, info = env.step(action)
        assert done

    def test_deterministic_reset_with_same_seed(self):
        config = SimulatorConfig(image_size=(224, 224))
        gen = get_target_generator_for_task("microsphere")
        env = MicroscopeEnvironment(config=config, target_generator=gen)
        obs1, info1 = env.reset(seed=999)
        obs2, info2 = env.reset(seed=999)
        assert np.array_equal(obs1, obs2)
        assert np.allclose(info1["target_position"], info2["target_position"])

    def test_step_before_reset_raises(self):
        config = SimulatorConfig(image_size=(224, 224))
        gen = get_target_generator_for_task("microsphere")
        env = MicroscopeEnvironment(config=config, target_generator=gen)
        with pytest.raises((AttributeError, TypeError)):
            env.step(np.array([1.0, 1.0]))

    def test_render_returns_image(self):
        config = SimulatorConfig(image_size=(224, 224))
        gen = get_target_generator_for_task("microsphere")
        env = MicroscopeEnvironment(config=config, target_generator=gen)
        env.reset(seed=42)
        rendered = env.render()
        assert rendered.shape == (224, 224, 3)
        assert rendered.dtype == np.uint8


# ------------------------------------------------------------------
# Target-specific environment tests
# ------------------------------------------------------------------

class TestMicrosphereEnvironment:
    CONFIG = SimulatorConfig(image_size=(224, 224))

    def test_env_reset_step(self):
        gen = get_target_generator_for_task("microsphere")
        env = MicroscopeEnvironment(config=self.CONFIG, target_generator=gen)
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert info["distance"] > 0
        action = np.array([5.0, -3.0], dtype=np.float64)
        obs2, reward, done, info2 = env.step(action)
        assert obs2 is not None
        assert info2["step"] == 1


# ------------------------------------------------------------------
# SingleImageBackground
# ------------------------------------------------------------------

class TestSingleImageBackground:
    def test_generate_shape(self):
        """测试背景生成器输出形状。"""
        # 需要存在背景图片才能运行此测试
        bg_path = Path("data/backgrounds_test/bg_00000.png")
        if not bg_path.exists():
            pytest.skip("背景图片不存在")

        bg = SingleImageBackground(image_path=bg_path)
        rng = np.random.default_rng(42)
        result = bg.generate((224, 224), rng)
        assert result.shape == (224, 224)
        assert result.dtype == np.uint8

    def test_brightness_adjustment(self):
        """测试亮度调整功能。"""
        bg_path = Path("data/backgrounds_test/bg_00000.png")
        if not bg_path.exists():
            pytest.skip("背景图片不存在")

        bg = SingleImageBackground(
            image_path=bg_path,
            brightness_range=(0.5, 0.5)  # 固定亮度
        )
        rng = np.random.default_rng(42)
        result = bg.generate((224, 224), rng)
        # 亮度调整应该保持在有效范围内
        assert result.min() >= 0
        assert result.max() <= 255

    def test_deterministic_with_seed(self):
        """测试相同种子生成相同结果。"""
        bg_path = Path("data/backgrounds_test/bg_00000.png")
        if not bg_path.exists():
            pytest.skip("背景图片不存在")

        bg = SingleImageBackground(image_path=bg_path)
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(12345)
        res1 = bg.generate((224, 224), rng1)
        res2 = bg.generate((224, 224), rng2)
        assert np.array_equal(res1, res2)

    def test_in_environment(self):
        """测试在环境中的使用。"""
        bg_path = Path("data/backgrounds_test/bg_00000.png")
        if not bg_path.exists():
            pytest.skip("背景图片不存在")

        gen = MicrosphereGenerator()
        config = SimulatorConfig(
            image_size=(224, 224),
            background_type="single_image",
            background_image=str(bg_path),
        )
        env = MicroscopeEnvironment(config=config, target_generator=gen)
        obs, info = env.reset(seed=42)
        assert obs.shape == (224, 224, 3)
        assert obs.dtype == np.uint8
