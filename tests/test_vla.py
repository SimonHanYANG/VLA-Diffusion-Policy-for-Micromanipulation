"""Tests for VLA model components: noise scheduler, config, metrics, diffusion unet."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

try:
    import torch
except (ImportError, OSError):
    pytest.skip("PyTorch not available or DLL loading failed", allow_module_level=True)

from src.utils.config import (
    SimulatorConfig,
    NoiseConfig,
    DomainRandConfig,
    TrainingConfig,
    DiffusionPolicyConfig,
    TaskConfig,
    _dict_to_dataclass,
    load_simulator_config,
    load_training_config,
    load_diffusion_policy_config,
    load_task_config,
)
from src.vla.noise_scheduler import NoiseScheduler
from src.vla.diffusion_unet import SinusoidalPosEmb, DownBlock1D, UpBlock1D, DiffusionUNet1D
from src.training.metrics import EpisodeMetrics, AggregateMetrics, compute_aggregate_metrics


# ------------------------------------------------------------------
# Config: YAML -> dataclass + tuple conversion
# ------------------------------------------------------------------

class TestConfigLoading:
    def test_load_simulator_config(self):
        cfg = load_simulator_config(Path("configs/simulator/default.yaml"))
        assert isinstance(cfg, SimulatorConfig)
        assert isinstance(cfg.image_size, tuple)
        assert len(cfg.image_size) == 2
        assert cfg.image_size == (224, 224)
        assert isinstance(cfg.noise, NoiseConfig)
        assert isinstance(cfg.domain_randomization, DomainRandConfig)

    def test_load_training_config(self):
        cfg = load_training_config(Path("configs/training/default.yaml"))
        assert isinstance(cfg, TrainingConfig)
        assert cfg.batch_size == 64
        assert cfg.learning_rate == 1e-4
        assert hasattr(cfg, "log_dir")
        assert cfg.log_dir == "data/runs"

    def test_load_diffusion_policy_config(self):
        cfg = load_diffusion_policy_config(Path("configs/model/diffusion_policy.yaml"))
        assert isinstance(cfg, DiffusionPolicyConfig)
        assert cfg.obs_horizon == 2
        assert cfg.action_dim == 2
        assert isinstance(cfg.condition_hidden_dims, list)
        assert isinstance(cfg.unet_dims, list)

    def test_load_all_task_configs(self):
        for task in ["microsphere", "yeast", "sperm_head", "sperm_tail"]:
            cfg = load_task_config(Path(f"configs/simulator/{task}.yaml"))
            assert isinstance(cfg, TaskConfig)
            assert cfg.name == task
            assert len(cfg.instruction) > 0
            assert len(cfg.target_generator) > 0

    def test_tuple_conversion(self):
        data = {
            "image_size": [300, 400],
            "um_per_pixel": 0.5,
            "noise": {"gaussian_std": 3.0},
            "domain_randomization": {"brightness_range": [0.3, 2.0]},
        }
        cfg = _dict_to_dataclass(SimulatorConfig, data)
        assert isinstance(cfg.image_size, tuple)
        assert cfg.image_size == (300, 400)
        assert isinstance(cfg.domain_randomization.brightness_range, tuple)
        assert cfg.domain_randomization.brightness_range == (0.3, 2.0)

    def test_extra_keys_ignored(self):
        data = {"image_size": (100, 100), "unknown_field": 42}
        cfg = _dict_to_dataclass(SimulatorConfig, data)
        assert cfg.image_size == (100, 100)


# ------------------------------------------------------------------
# NoiseScheduler
# ------------------------------------------------------------------

class TestNoiseScheduler:
    def test_cosine_schedule_init(self):
        ns = NoiseScheduler(num_train_steps=100, num_ddim_steps=10, schedule="cosine")
        assert ns.betas.shape == (100,)
        assert ns.alphas_cumprod.shape == (100,)
        # Betas should increase monotonically
        assert torch.all(ns.betas[1:] >= ns.betas[:-1])

    def test_linear_schedule_init(self):
        ns = NoiseScheduler(num_train_steps=100, num_ddim_steps=10, schedule="linear")
        assert ns.betas.shape == (100,)

    def test_unknown_schedule_raises(self):
        with pytest.raises(ValueError, match="Unknown beta schedule"):
            NoiseScheduler(schedule="invalid")

    def test_add_noise_shape(self):
        ns = NoiseScheduler(num_train_steps=100)
        x0 = torch.randn(4, 2, 10)
        noise = torch.randn(4, 2, 10)
        t = torch.randint(0, 100, (4,))
        noisy = ns.add_noise(x0, noise, t)
        assert noisy.shape == x0.shape

    def test_ddim_timesteps_order(self):
        ns = NoiseScheduler(num_train_steps=100, num_ddim_steps=10)
        ts = ns.get_ddim_timesteps(torch.device("cpu"))
        # Should be strictly descending
        assert torch.all(ts[:-1] > ts[1:])
        # Last timestep should be 0
        assert ts[-1] == 0

    def test_ddim_timesteps_validation(self):
        ns = NoiseScheduler(num_train_steps=100, num_ddim_steps=0)
        with pytest.raises(ValueError, match="positive"):
            ns.get_ddim_timesteps(torch.device("cpu"))

        ns2 = NoiseScheduler(num_train_steps=10, num_ddim_steps=20)
        with pytest.raises(ValueError, match="cannot exceed"):
            ns2.get_ddim_timesteps(torch.device("cpu"))

    def test_ddim_step_shape(self):
        ns = NoiseScheduler(num_train_steps=100, num_ddim_steps=10)
        x_t = torch.randn(2, 2, 10)
        noise_pred = torch.randn(2, 2, 10)
        t, t_prev = 99, 89
        x_prev = ns.ddim_step(x_t, noise_pred, t, t_prev)
        assert x_prev.shape == x_t.shape

    def test_cosine_betas_in_range(self):
        betas = NoiseScheduler._cosine_beta_schedule(100)
        assert torch.all(betas >= 0.0001)
        assert torch.all(betas <= 0.02)


# ------------------------------------------------------------------
# SinusoidalPosEmb
# ------------------------------------------------------------------

class TestSinusoidalPosEmb:
    def test_standard_dim(self):
        spe = SinusoidalPosEmb(128)
        t = torch.tensor([0, 50, 99])
        emb = spe(t)
        assert emb.shape == (3, 128)

    def test_small_dim(self):
        for dim in [0, 1, 2]:
            spe = SinusoidalPosEmb(dim)
            t = torch.tensor([0, 10])
            emb = spe(t)
            assert emb.shape == (2, dim)


# ------------------------------------------------------------------
# DiffusionUNet1D shape propagation
# ------------------------------------------------------------------

class TestDiffusionUNet1D:
    def test_forward_shape(self):
        unet = DiffusionUNet1D(
            action_dim=2,
            horizon=10,
            cond_dim=256,
            dims=[64, 128, 256],
        )
        x = torch.randn(4, 2, 10)
        t = torch.tensor([0, 50, 99, 99])
        cond = torch.randn(4, 256)
        out = unet(x, t, cond)
        assert out.shape == x.shape

    def test_different_horizon(self):
        unet = DiffusionUNet1D(
            action_dim=2,
            horizon=16,
            cond_dim=256,
            dims=[64, 128],
        )
        x = torch.randn(2, 2, 16)
        t = torch.tensor([0, 50])
        cond = torch.randn(2, 256)
        out = unet(x, t, cond)
        assert out.shape == x.shape


# ------------------------------------------------------------------
# DownBlock1D residual always produces correct shape
# ------------------------------------------------------------------

class TestDownBlock1D:
    def test_residual_shape_match_different_dims(self):
        block = DownBlock1D(64, 128, cond_dim=256)
        x = torch.randn(2, 64, 20)
        cond = torch.randn(2, 256)
        out = block(x, cond)
        assert out.shape == (2, 128, 10)  # stride 2 halves spatial

    def test_residual_shape_match_same_dims(self):
        # This was the bug case: in_dim == out_dim
        block = DownBlock1D(64, 64, cond_dim=256)
        x = torch.randn(2, 64, 20)
        cond = torch.randn(2, 256)
        out = block(x, cond)
        assert out.shape == (2, 64, 10)  # spatial halved, channels same


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

class TestMetrics:
    def test_smoothness_zero_for_straight_line(self):
        actions = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        smoothness = EpisodeMetrics.compute_smoothness(actions)
        assert smoothness == 0.0 or smoothness < 1e-6

    def test_smoothness_nonzero_for_turns(self):
        actions = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
        smoothness = EpisodeMetrics.compute_smoothness(actions)
        assert smoothness > 0.01

    def test_smoothness_empty_actions(self):
        actions = np.zeros((0, 2))
        smoothness = EpisodeMetrics.compute_smoothness(actions)
        assert smoothness == 0.0

    def test_smoothness_single_action(self):
        actions = np.array([[1.0, 2.0]])
        smoothness = EpisodeMetrics.compute_smoothness(actions)
        assert smoothness == 0.0

    def test_aggregate_metrics(self):
        eps = [
            EpisodeMetrics(
                success=True, final_distance=1.0, trajectory_length=50,
                mean_step_distance=5.0, trajectory_smoothness=0.1, task_name="test",
            ),
            EpisodeMetrics(
                success=False, final_distance=10.0, trajectory_length=200,
                mean_step_distance=3.0, trajectory_smoothness=0.3, task_name="test",
            ),
        ]
        agg = compute_aggregate_metrics(eps)
        assert agg.success_rate == 0.5
        assert agg.num_episodes == 2
        assert agg.mean_final_distance == 5.5

    def test_aggregate_all_success(self):
        eps = [
            EpisodeMetrics(
                success=True, final_distance=0.5, trajectory_length=100,
                mean_step_distance=2.0, trajectory_smoothness=0.05, task_name="test",
            )
            for _ in range(5)
        ]
        agg = compute_aggregate_metrics(eps)
        assert agg.success_rate == 1.0
