from typing import Tuple

import numpy as np

from src.interfaces.stage_interface import StageInterface


class StageSimulator(StageInterface):
    """Simulated microscope stage. Mirrors the real Nikon Ti2E stage API.

    Internally tracks position in micrometers. The MicroscopeEnvironment
    translates between stage displacements and pixel-space object translations
    using a configurable um_per_pixel scale factor with +/- 10% random variation
    to simulate calibration error.
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-25000.0, 25000.0),
        y_range: Tuple[float, float] = (-25000.0, 25000.0),
        z_range: Tuple[float, float] = (0.0, 5000.0),
    ):
        self._x: float = 0.0
        self._y: float = 0.0
        self._z: float = 0.0
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self._connected = False

    def connect(self) -> None:
        self._connected = True

    def dispose(self) -> None:
        self._connected = False

    def _check_connected(self) -> None:
        if not self._connected:
            import warnings
            warnings.warn("Stage not connected. Call connect() before moving.", stacklevel=3)

    def move_xy_to_absolute(self, x: float, y: float) -> None:
        self._check_connected()
        self._x = float(np.clip(x, *self.x_range))
        self._y = float(np.clip(y, *self.y_range))

    def move_xy_relative(self, x_delta: float, y_delta: float) -> None:
        self.move_xy_to_absolute(self._x + x_delta, self._y + y_delta)

    def get_xy_position(self) -> Tuple[float, float]:
        return (self._x, self._y)

    def move_z_to_absolute(self, z: float) -> None:
        self._z = float(np.clip(z, *self.z_range))

    def move_z_relative(self, z_delta: float) -> None:
        self.move_z_to_absolute(self._z + z_delta)

    def get_z_position(self) -> float:
        return self._z

    def reset_position(self) -> None:
        self._x = 0.0
        self._y = 0.0
        self._z = 0.0
