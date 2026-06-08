from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TargetRender:
    mask: np.ndarray
    reference_point: Tuple[float, float]
    label: str


class TargetGenerator(ABC):
    @abstractmethod
    def generate(self, image_size: Tuple[int, int], rng: np.random.Generator) -> TargetRender:
        ...


class MicrosphereGenerator(TargetGenerator):
    """Rigid circle with solid or ring texture, radius 15-25 px."""

    def __init__(self, radius_min: float = 15.0, radius_max: float = 25.0,
                 ring_prob: float = 0.3, ring_thickness: float = 3.0):
        self.radius_min = radius_min
        self.radius_max = radius_max
        self.ring_prob = ring_prob
        self.ring_thickness = ring_thickness

    def generate(self, image_size: Tuple[int, int], rng: np.random.Generator) -> TargetRender:
        h, w = image_size
        radius = rng.uniform(self.radius_min, self.radius_max)
        cx, cy = w / 2.0, h / 2.0

        y_grid, x_grid = np.ogrid[:h, :w]
        dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)

        if rng.random() < self.ring_prob:
            thickness = self.ring_thickness * rng.uniform(0.8, 1.2)
            mask = ((dist >= radius - thickness) & (dist <= radius)).astype(np.uint8) * 255
        else:
            sigma = radius * 0.05
            mask = (255 * (1.0 - np.clip((dist - radius + sigma) / (2 * sigma), 0.0, 1.0))).astype(np.uint8)
            mask[dist <= radius - sigma] = 255

        return TargetRender(mask=mask, reference_point=(cx, cy), label="microsphere")


class YeastGenerator(TargetGenerator):
    """Ellipse with sinusoidal bumps on perimeter, major axis 20-35 px."""

    def __init__(self, major_min: float = 20.0, major_max: float = 35.0,
                 aspect_ratio_range: Tuple[float, float] = (0.6, 0.9),
                 num_bumps: Tuple[int, int] = (3, 7), bump_amp: float = 0.06):
        self.major_min = major_min
        self.major_max = major_max
        self.aspect_ratio_range = aspect_ratio_range
        self.num_bumps = num_bumps
        self.bump_amp = bump_amp

    def generate(self, image_size: Tuple[int, int], rng: np.random.Generator) -> TargetRender:
        h, w = image_size
        cx, cy = w / 2.0, h / 2.0
        major = rng.uniform(self.major_min, self.major_max)
        aspect = rng.uniform(*self.aspect_ratio_range)
        minor = major * aspect
        angle = rng.uniform(0, 2 * np.pi)
        n_bumps = rng.integers(*self.num_bumps)

        y_grid, x_grid = np.ogrid[:h, :w]
        dx = x_grid - cx
        dy = y_grid - cy
        xr = dx * np.cos(angle) + dy * np.sin(angle)
        yr = -dx * np.sin(angle) + dy * np.cos(angle)

        # Ellipse radial distance
        theta = np.arctan2(yr, xr)
        r_ellipse = (major * minor) / np.sqrt((minor * np.cos(theta)) ** 2 + (major * np.sin(theta)) ** 2)

        # Add bumps
        bump_factor = 1.0
        for i in range(n_bumps):
            fi = rng.uniform(3, 8)
            phi_i = rng.uniform(0, 2 * np.pi)
            amp_i = rng.uniform(0, self.bump_amp)
            bump_factor += amp_i * np.sin(fi * theta + phi_i)

        r_target = r_ellipse * bump_factor
        r_actual = np.sqrt(xr ** 2 + yr ** 2)

        sigma = 1.0
        mask = (255 * (1.0 - np.clip((r_actual - r_target + sigma) / (2 * sigma), 0.0, 1.0))).astype(np.uint8)
        mask[r_actual <= r_target - sigma] = 255

        return TargetRender(mask=mask, reference_point=(cx, cy), label="yeast")


class SpermHeadGenerator(TargetGenerator):
    """Ellipse with tapered tip, major axis 18-28 px."""

    def __init__(self, major_min: float = 18.0, major_max: float = 28.0,
                 aspect_ratio_range: Tuple[float, float] = (0.4, 0.7),
                 tip_sharpness_range: Tuple[float, float] = (2.0, 5.0)):
        self.major_min = major_min
        self.major_max = major_max
        self.aspect_ratio_range = aspect_ratio_range
        self.tip_sharpness_range = tip_sharpness_range

    def generate(self, image_size: Tuple[int, int], rng: np.random.Generator) -> TargetRender:
        h, w = image_size
        cx, cy = w / 2.0, h / 2.0
        major = rng.uniform(self.major_min, self.major_max)
        aspect = rng.uniform(*self.aspect_ratio_range)
        minor = major * aspect
        angle = rng.uniform(0, 2 * np.pi)
        sharpness = rng.uniform(*self.tip_sharpness_range)

        y_grid, x_grid = np.ogrid[:h, :w]
        dx = x_grid - cx
        dy = y_grid - cy
        xr = dx * np.cos(angle) + dy * np.sin(angle)
        yr = -dx * np.sin(angle) + dy * np.cos(angle)

        theta = np.arctan2(yr, xr)
        r_ellipse = (major * minor) / np.sqrt((minor * np.cos(theta)) ** 2 + (major * np.sin(theta)) ** 2)

        # Taper the tip at theta near pi (pointed end)
        taper = np.ones_like(theta)
        tip_region = np.abs(theta - np.pi) < np.pi / 3
        tip_angle = np.abs(theta[tip_region] - np.pi)
        taper[tip_region] *= (tip_angle / (np.pi / 3)) ** sharpness

        r_target = r_ellipse * taper
        r_actual = np.sqrt(xr ** 2 + yr ** 2)

        sigma = 1.0
        mask = (255 * (1.0 - np.clip((r_actual - r_target + sigma) / (2 * sigma), 0.0, 1.0))).astype(np.uint8)
        mask[r_actual <= r_target - sigma] = 255

        return TargetRender(mask=mask, reference_point=(cx, cy), label="sperm_head")


class SpermTailGenerator(TargetGenerator):
    """Cubic Bezier curve tail. Reference point = tip (endpoint) of the curve."""

    def __init__(self, line_width: Tuple[float, float] = (1.0, 2.5),
                 curve_length_range: Tuple[float, float] = (40.0, 100.0)):
        self.line_width = line_width
        self.curve_length_range = curve_length_range

    def generate(self, image_size: Tuple[int, int], rng: np.random.Generator) -> TargetRender:
        h, w = image_size
        cx, cy = w / 2.0, h / 2.0

        curve_len = rng.uniform(*self.curve_length_range)
        angle = rng.uniform(0, 2 * np.pi)

        # Generate 4 control points for cubic Bezier
        p0 = np.array([cx, cy])
        p3 = np.array([cx + curve_len * np.cos(angle), cy + curve_len * np.sin(angle)])

        # Mid control points with random offsets for natural variation
        mid = (p0 + p3) / 2
        perp = np.array([-np.sin(angle), np.cos(angle)]) * curve_len * 0.3
        p1 = p0 * 0.7 + p3 * 0.3 + perp * rng.uniform(-1, 1) + rng.uniform(-5, 5, 2)
        p2 = p0 * 0.3 + p3 * 0.7 + perp * rng.uniform(-1, 1) + rng.uniform(-5, 5, 2)

        # Render curve by sampling Bezier
        mask = np.zeros((h, w), dtype=np.uint8)
        num_samples = 200
        t_vals = np.linspace(0, 1, num_samples)
        points = np.zeros((num_samples, 2))

        for i, t in enumerate(t_vals):
            t_inv = 1 - t
            pt = t_inv**3 * p0 + 3 * t_inv**2 * t * p1 + 3 * t_inv * t**2 * p2 + t**3 * p3
            points[i] = pt

        line_w = rng.uniform(*self.line_width)
        for i in range(num_samples):
            px, py = int(round(points[i, 0])), int(round(points[i, 1]))
            if 0 <= px < w and 0 <= py < h:
                cv2_circle_fill(mask, (px, py), int(np.ceil(line_w / 2)))

        tip_point = (float(points[-1, 0]), float(points[-1, 1]))

        # Attach a small head at the base (p0 side) for realism
        head_radius = rng.uniform(6, 10)
        head_mask = np.zeros((h, w), dtype=np.uint8)
        cv2_circle_fill(head_mask, (int(p0[0]), int(p0[1])), int(head_radius))
        # Blend head into mask
        mask = np.maximum(mask, head_mask)

        return TargetRender(mask=mask, reference_point=tip_point, label="sperm_tail")


def cv2_circle_fill(img: np.ndarray, center: Tuple[int, int], radius: int) -> None:
    """Fill a circle on a numpy array without cv2 dependency for pure array ops."""
    h, w = img.shape
    cx, cy = center
    y_grid, x_grid = np.ogrid[:h, :w]
    dist = np.sqrt((x_grid - cx) ** 2 + (y_grid - cy) ** 2)
    img[dist <= radius] = 255


TASK_TO_GENERATOR = {
    "microsphere": MicrosphereGenerator,
    "yeast": YeastGenerator,
    "sperm_head": SpermHeadGenerator,
    "sperm_tail": SpermTailGenerator,
}

GENERATOR_REGISTRY = {
    "MicrosphereGenerator": MicrosphereGenerator,
    "YeastGenerator": YeastGenerator,
    "SpermHeadGenerator": SpermHeadGenerator,
    "SpermTailGenerator": SpermTailGenerator,
}


def get_target_generator_for_task(task_name: str) -> TargetGenerator:
    """Create a default target generator for a named task.

    Args:
        task_name: Lowercase task name, e.g. "microsphere", "yeast".

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
