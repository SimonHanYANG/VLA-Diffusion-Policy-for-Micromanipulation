import numpy as np


def _fade(t: np.ndarray) -> np.ndarray:
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _lerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    return a + t * (b - a)


def _gradient(h: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute dot product between pseudorandom gradient vectors and offset vectors."""
    h = h & 3
    u = np.where(h < 2, x, y)
    v = np.where(h < 2, y, x)
    return np.where((h & 1) == 0, u, -u) + np.where((h & 2) == 0, v, -v)


def generate_perlin_noise(
    width: int,
    height: int,
    scale: float = 50.0,
    octaves: int = 4,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate 2D Perlin noise texture.

    Args:
        width, height: output image dimensions.
        scale: base spatial scale (larger = coarser features).
        octaves: number of noise octaves to sum.
        persistence: amplitude multiplier per octave.
        lacunarity: frequency multiplier per octave.
        seed: optional random seed for reproducibility.

    Returns:
        (height, width) float64 array in [0, 1].
    """
    rng = np.random.default_rng(seed)

    # Generate permutation table
    p = np.arange(256, dtype=int)
    rng.shuffle(p)
    p = np.concatenate([p, p])

    noise = np.zeros((height, width), dtype=np.float64)
    amplitude = 1.0
    total_amplitude = 0.0
    freq = 1.0

    for _ in range(octaves):
        scaled_w = max(1, int(width * freq / scale))
        scaled_h = max(1, int(height * freq / scale))

        # Generate grid coordinates
        xi = np.linspace(0, scaled_w, width, endpoint=False)
        yi = np.linspace(0, scaled_h, height, endpoint=False)
        x_grid, y_grid = np.meshgrid(xi, yi)

        xi0 = np.floor(x_grid).astype(int) % 256
        yi0 = np.floor(y_grid).astype(int) % 256
        xi1 = (xi0 + 1) % 256
        yi1 = (yi0 + 1) % 256

        sx = _fade(x_grid - np.floor(x_grid))
        sy = _fade(y_grid - np.floor(y_grid))

        # Hash corners
        n00 = p[p[xi0] + yi0]
        n10 = p[p[xi1] + yi0]
        n01 = p[p[xi0] + yi1]
        n11 = p[p[xi1] + yi1]

        # Gradients
        g00 = _gradient(n00, x_grid - np.floor(x_grid), y_grid - np.floor(y_grid))
        g10 = _gradient(n10, x_grid - np.floor(x_grid) - 1, y_grid - np.floor(y_grid))
        g01 = _gradient(n01, x_grid - np.floor(x_grid), y_grid - np.floor(y_grid) - 1)
        g11 = _gradient(n11, x_grid - np.floor(x_grid) - 1, y_grid - np.floor(y_grid) - 1)

        nx0 = _lerp(g00, g10, sx)
        nx1 = _lerp(g01, g11, sx)
        octave_noise = _lerp(nx0, nx1, sy)

        noise += amplitude * octave_noise
        total_amplitude += amplitude
        amplitude *= persistence
        freq *= lacunarity

    noise /= total_amplitude
    return (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
