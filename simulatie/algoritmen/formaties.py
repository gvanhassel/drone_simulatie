"""Zwermformaties: V-formatie, cirkel en grid."""

from __future__ import annotations

import math
from uuid import UUID

import numpy as np


class VFormatie:
    """V-patroon rondom een centrum, gericht in vliegrichting."""

    def apply(self, drones: list, center: np.ndarray, heading: float = 0.0) -> dict[UUID, np.ndarray]:
        """Geeft doel-posities terug per drone."""
        n = len(drones)
        if n == 0:
            return {}
        spacing = 40.0
        result = {}
        for i, drone in enumerate(drones):
            if i == 0:
                result[drone.id] = center.copy()
            else:
                side = 1 if i % 2 == 1 else -1
                row = (i + 1) // 2
                dx = -row * spacing * math.cos(heading) + side * row * spacing * math.sin(heading) * 0.5
                dy = -row * spacing * math.sin(heading) - side * row * spacing * math.cos(heading) * 0.5
                result[drone.id] = center + np.array([dx, dy], dtype=np.float32)
        return result


class CirkelFormatie:
    """Drones gelijkmatig verdeeld op een cirkel."""

    def __init__(self, radius: float = 120.0) -> None:
        self.radius = radius

    def apply(self, drones: list, center: np.ndarray, heading: float = 0.0) -> dict[UUID, np.ndarray]:
        n = len(drones)
        if n == 0:
            return {}
        result = {}
        for i, drone in enumerate(drones):
            angle = 2 * math.pi * i / n
            x = center[0] + self.radius * math.cos(angle)
            y = center[1] + self.radius * math.sin(angle)
            result[drone.id] = np.array([x, y], dtype=np.float32)
        return result


class GridFormatie:
    """Rechthoekig grid rondom centrum."""

    def __init__(self, spacing: float = 50.0) -> None:
        self.spacing = spacing

    def apply(self, drones: list, center: np.ndarray, heading: float = 0.0) -> dict[UUID, np.ndarray]:
        n = len(drones)
        if n == 0:
            return {}
        cols = math.ceil(math.sqrt(n))
        result = {}
        for i, drone in enumerate(drones):
            row = i // cols
            col = i % cols
            total_cols = min(cols, n - row * cols)
            # Centreer de rij
            offset_x = (col - (total_cols - 1) / 2.0) * self.spacing
            offset_y = (row - (math.ceil(n / cols) - 1) / 2.0) * self.spacing
            result[drone.id] = center + np.array([offset_x, offset_y], dtype=np.float32)
        return result


class FormatieBeheerder:
    """Beheert de actieve formatie en berekent doel-posities."""

    FORMATIES = {
        0: None,            # vrije flocking
        1: VFormatie,
        2: CirkelFormatie,
        3: GridFormatie,
    }

    def __init__(self) -> None:
        self._mode: int = 0
        self._formatie_instanties = {
            1: VFormatie(),
            2: CirkelFormatie(),
            3: GridFormatie(),
        }

    @property
    def mode(self) -> int:
        return self._mode

    def set_mode(self, mode: int) -> None:
        self._mode = mode if mode in self.FORMATIES else 0

    def get_doel_posities(
        self,
        drones: list,
        center: np.ndarray,
        heading: float = 0.0,
    ) -> dict[UUID, np.ndarray] | None:
        """
        Geeft doel-posities terug als een formatie actief is.
        Geeft None terug bij vrije flocking (mode=0).
        """
        if self._mode == 0 or not drones:
            return None
        formatie = self._formatie_instanties.get(self._mode)
        if formatie is None:
            return None
        return formatie.apply(drones, center, heading)
