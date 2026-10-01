from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf as _scipy_erf


@dataclass(frozen=True)
class FocusPoint:
    pos: float
    axis: str
    factor: float = 2.0
    extent: float = 0.1

    def __post_init__(self) -> None:
        axis = self.axis.lower()
        if not 0.0 <= self.pos <= 1.0:
            raise ValueError("pos must be within [0, 1]")
        if axis not in {"x", "y"}:
            raise ValueError("axis must be 'x' or 'y'")
        if not np.isfinite(self.factor) or self.factor <= 0.0:
            raise ValueError("factor must be positive")
        if not np.isfinite(self.extent) or self.extent <= 0.0:
            raise ValueError("extent must be positive")
        object.__setattr__(self, "axis", axis)

    def _reposition_point(self, point: np.ndarray) -> np.ndarray:
        alpha = 1.0 - 1.0 / self.factor
        erf = _scipy_erf((point - self.pos) / self.extent)
        return point - 0.5 * (np.sqrt(np.pi) * self.extent * alpha * erf)

    def _focus_array(self, array: np.ndarray) -> np.ndarray:
        f0 = self._reposition_point(np.asarray(0.0))
        f1 = self._reposition_point(np.asarray(1.0))
        return (self._reposition_point(array) - f0) / (f1 - f0)

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if np.any((x < 0.0) | (x > 1.0)):
            raise ValueError("x must be within [0, 1]")
        if np.any((y < 0.0) | (y > 1.0)):
            raise ValueError("y must be within [0, 1]")
        if self.axis == "x":
            return self._focus_array(x), y
        return x, self._focus_array(y)


@dataclass
class Focus:
    points: list[FocusPoint] = field(default_factory=list)

    def add_focus(
        self,
        pos: float,
        axis: str,
        factor: float = 2.0,
        extent: float = 0.1,
    ) -> None:
        self.points.append(FocusPoint(pos=pos, axis=axis, factor=factor, extent=extent))

    def __call__(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xf = np.asarray(x, dtype=float)
        yf = np.asarray(y, dtype=float)
        for point in self.points:
            xf, yf = point(xf, yf)
        return xf, yf

    def focused_coordinates(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        ny, nx = shape
        xi = np.linspace(0.0, 1.0, nx)
        eta = np.linspace(0.0, 1.0, ny)
        XI, ETA = np.meshgrid(xi, eta)
        return self(XI, ETA)
