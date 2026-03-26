"""Camera met pan en zoom voor de simulatieweergave."""

from __future__ import annotations

import numpy as np


class Camera:
    """
    2D camera met pan en zoom.
    Zet wereldcoördinaten om naar schermcoördinaten en omgekeerd.
    """

    def __init__(self, screen_w: int, screen_h: int) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self._zoom: float = 1.0
        self._offset: np.ndarray = np.zeros(2, dtype=np.float64)  # wereld-offset
        self._min_zoom: float = 0.25
        self._max_zoom: float = 4.0

    @property
    def zoom(self) -> float:
        return self._zoom

    @property
    def offset(self) -> np.ndarray:
        return self._offset.copy()

    def world_to_screen(self, world_pos: np.ndarray) -> np.ndarray:
        """Wereldcoördinaten → schermcoördinaten."""
        return ((world_pos - self._offset) * self._zoom).astype(np.float32)

    def screen_to_world(self, screen_pos: np.ndarray) -> np.ndarray:
        """Schermcoördinaten → wereldcoördinaten."""
        return (screen_pos / self._zoom + self._offset).astype(np.float32)

    def pan(self, dx: float, dy: float) -> None:
        """Verschuif de camera met dx, dy in schermcoördinaten."""
        self._offset += np.array([dx, dy], dtype=np.float64) / self._zoom

    def zoom_by(self, delta: float) -> None:
        """Pas zoom aan met delta (positief = inzoomen)."""
        new_zoom = self._zoom + delta
        self._zoom = float(np.clip(new_zoom, self._min_zoom, self._max_zoom))

    def reset(self) -> None:
        """Reset camera naar standaard positie."""
        self._zoom = 1.0
        self._offset = np.zeros(2, dtype=np.float64)
