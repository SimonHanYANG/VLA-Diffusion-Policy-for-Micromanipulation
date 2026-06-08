from abc import ABC, abstractmethod
from typing import Tuple


class StageInterface(ABC):
    """Abstract interface matching Nikon Ti2E stage API.

    All coordinates in micrometers. Positive X = right, positive Y = up.
    Both StageSimulator and the real NikonTi2EStage implement this interface
    so the environment can swap between them with a one-line change.
    """

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the stage hardware."""
        ...

    @abstractmethod
    def dispose(self) -> None:
        """Release hardware resources."""
        ...

    @abstractmethod
    def move_xy_to_absolute(self, x: float, y: float) -> None:
        """Move stage to absolute (x, y) position in micrometers."""
        ...

    @abstractmethod
    def move_xy_relative(self, x_delta: float, y_delta: float) -> None:
        """Move stage by relative offset in micrometers."""
        ...

    @abstractmethod
    def get_xy_position(self) -> Tuple[float, float]:
        """Return current absolute (x, y) position in micrometers."""
        ...

    @abstractmethod
    def move_z_to_absolute(self, z: float) -> None:
        """Move focus to absolute z in micrometers."""
        ...

    @abstractmethod
    def move_z_relative(self, z_delta: float) -> None:
        """Move focus by relative offset in micrometers."""
        ...

    @abstractmethod
    def get_z_position(self) -> float:
        """Return current absolute z position in micrometers."""
        ...
