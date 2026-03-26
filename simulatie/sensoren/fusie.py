"""
SensorFusion: combineert alle drone-sensorreadings tot één 2D heatmap-array.

Gebaseerd op Soorati et al. (2021): gecombineerde heatmap-weergave is
empirisch superieur voor enkelvoudige operatorbesturing van drone-zwermen.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from simulatie.config import SensorConfig
from simulatie.sensoren.proximity import ProximitySensor
from simulatie.sensoren.sensor import SensorReading


class SensorFusion:
    """
    Accumuleert sensorreadings van alle drones tot een 2D numpy array.

    Pipeline per frame:
        1. Reset accumulator naar nul
        2. Projecteer elke reading op gridcel
        3. Decay: accumulator *= decay_rate (temporeel geheugen)
        4. Gaussian blur voor visuele smoothing
        5. Normaliseer naar [0, 1]
    """

    def __init__(self, world_w: int, world_h: int, config: SensorConfig | None = None) -> None:
        cfg = config or SensorConfig()
        self.config = cfg
        self.world_w = world_w
        self.world_h = world_h

        self.grid_w = world_w // cfg.grid_resolution
        self.grid_h = world_h // cfg.grid_resolution

        # Persistente accumulator — behoudt temporele informatie via decay
        self._accumulator: np.ndarray = np.zeros(
            (self.grid_h, self.grid_w), dtype=np.float32
        )
        self._sensor = ProximitySensor()

    # ── Publieke API ──────────────────────────────────────────────────────────

    def fuse(self, drones: list) -> np.ndarray:
        """
        Scan alle drones en produceer een genormaliseerde heatmap [0, 1].

        Returns:
            np.ndarray van shape (grid_h, grid_w), float32, waarden in [0, 1].
        """
        # Stap 1: nieuwe frame-accumulatie op nul
        frame_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        # Stap 2: scan alle drones en projecteer readings
        for drone in drones:
            readings = self._sensor.scan(drone, drone._world_ref)  # type: ignore[attr-defined]
            self._project_readings(readings, frame_grid)

        # Stap 3: decay van vorige frames + toevoegen nieuwe frame
        self._accumulator *= self.config.decay_rate
        self._accumulator += frame_grid

        # Stap 4: Gaussian blur
        blurred = gaussian_filter(self._accumulator, sigma=self.config.blur_sigma)

        # Stap 5: normaliseer naar [0, 1]
        max_val = float(blurred.max())
        if max_val > 1e-8:
            normalized = blurred / max_val
        else:
            normalized = blurred.copy()

        return normalized.astype(np.float32)

    def fuse_with_world(self, drones: list, world) -> np.ndarray:
        """
        Alternatieve API waarbij de world expliciet wordt meegegeven.
        Gebruik deze in de simulatieloop.
        """
        frame_grid = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)

        for drone in drones:
            readings = self._sensor.scan(drone, world)
            drone.sensor_readings = readings
            self._project_readings(readings, frame_grid)

        self._accumulator *= self.config.decay_rate
        self._accumulator += frame_grid

        blurred = gaussian_filter(self._accumulator, sigma=self.config.blur_sigma)

        max_val = float(blurred.max())
        if max_val > 1e-8:
            normalized = blurred / max_val
        else:
            normalized = blurred.copy()

        return normalized.astype(np.float32)

    def reset(self) -> None:
        """Reset de accumulator (bijv. bij operator-verzoek via 'R'-toets)."""
        self._accumulator.fill(0.0)

    def world_to_grid(self, pos: np.ndarray) -> tuple[int, int]:
        """Zet wereldcoördinaten om naar gridcel-indices (row, col)."""
        col = int(np.clip(pos[0], 0, self.world_w - 1)) // self.config.grid_resolution
        row = int(np.clip(pos[1], 0, self.world_h - 1)) // self.config.grid_resolution
        col = min(col, self.grid_w - 1)
        row = min(row, self.grid_h - 1)
        return (row, col)

    # ── Interne methoden ──────────────────────────────────────────────────────

    def _project_readings(
        self, readings: list[SensorReading], grid: np.ndarray
    ) -> None:
        """Voeg alle readings toe aan het framegrid."""
        for reading in readings:
            row, col = self.world_to_grid(reading.absolute_position)
            grid[row, col] += reading.confidence
