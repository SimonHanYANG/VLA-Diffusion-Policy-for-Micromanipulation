from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple, get_args, get_origin

import yaml


@dataclass
class NoiseConfig:
    gaussian_std: float = 3.0
    salt_pepper_prob: float = 0.01
    flicker_prob: float = 0.1
    flicker_intensity_min: float = 0.7
    flicker_intensity_max: float = 1.3
    motion_blur_kernel_max: int = 5
    motion_blur_prob: float = 0.05
    fixed_pattern_prob: float = 0.3
    fixed_pattern_strength: float = 1.5


@dataclass
class DomainRandConfig:
    brightness_range: Tuple[float, float] = (0.5, 1.5)
    contrast_range: Tuple[float, float] = (0.5, 1.5)
    blur_sigma_max: float = 2.0
    blur_prob: float = 0.3


@dataclass
class OpticsConfig:
    """Optical effects configuration for realistic microscope simulation."""
    vignette_prob: float = 0.7
    vignette_strength_range: Tuple[float, float] = (0.1, 0.4)
    vignette_power: float = 2.0
    psf_prob: float = 0.5
    psf_sigma_range: Tuple[float, float] = (0.3, 1.2)
    chromatic_prob: float = 0.3
    chromatic_shift_range: Tuple[float, float] = (0.3, 1.5)


@dataclass
class SimulatorConfig:
    image_size: Tuple[int, int] = (224, 224)
    um_per_pixel: float = 0.5
    um_per_pixel_noise: float = 0.1
    max_steps_per_episode: int = 200
    success_tolerance_px: float = 2.0
    start_distance_min: float = 50.0
    start_distance_max: float = 200.0
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    domain_randomization: DomainRandConfig = field(default_factory=DomainRandConfig)
    optics: OpticsConfig = field(default_factory=OpticsConfig)


@dataclass
class TaskConfig:
    name: str = ""
    instruction: str = ""
    target_generator: str = ""
    target_params: dict = field(default_factory=dict)
    pid_kp: float = 0.5
    pid_ki: float = 0.01
    pid_kd: float = 0.1


@dataclass
class DiffusionPolicyConfig:
    obs_horizon: int = 2
    pred_horizon: int = 10
    action_dim: int = 2
    visual_backbone: str = "resnet18"
    visual_output_dim: int = 256
    text_model: str = "openai/clip-vit-base-patch32"
    text_dim: int = 512
    condition_hidden_dims: list = field(default_factory=lambda: [512, 256])
    condition_output_dim: int = 256
    unet_dims: list = field(default_factory=lambda: [64, 128, 256])
    num_diffusion_steps: int = 100
    num_ddim_steps: int = 10
    beta_schedule: str = "cosine"


@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 1e-6
    num_epochs: int = 500
    warmup_epochs: int = 10
    grad_accum_steps: int = 1
    val_split: float = 0.1
    val_freq_epochs: int = 10
    save_freq_epochs: int = 50
    early_stopping_patience: int = 100
    log_dir: str = "data/runs"


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _dict_to_dataclass(cls: type, data: dict) -> Any:
    from dataclasses import fields
    field_names = {f.name for f in fields(cls) if f.init}
    filtered = {k: v for k, v in data.items() if k in field_names}
    for fld in fields(cls):
        if fld.name not in filtered:
            continue
        val = filtered[fld.name]
        # Nested dataclass
        if _is_dataclass_type(fld.type):
            filtered[fld.name] = _dict_to_dataclass(fld.type, val)
        # YAML list -> tuple for Tuple[...] fields
        elif get_origin(fld.type) is tuple and isinstance(val, list):
            filtered[fld.name] = tuple(val)
    return cls(**filtered)


def _is_dataclass_type(tp: type) -> bool:
    import dataclasses
    return dataclasses.is_dataclass(tp)


def load_simulator_config(path: Path) -> SimulatorConfig:
    return _dict_to_dataclass(SimulatorConfig, load_yaml(path))


def load_task_config(path: Path) -> TaskConfig:
    return _dict_to_dataclass(TaskConfig, load_yaml(path))


def load_diffusion_policy_config(path: Path) -> DiffusionPolicyConfig:
    return _dict_to_dataclass(DiffusionPolicyConfig, load_yaml(path))


def load_training_config(path: Path) -> TrainingConfig:
    return _dict_to_dataclass(TrainingConfig, load_yaml(path))
