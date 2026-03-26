"""ProximitySensor: detecteert entiteiten binnen bereik en gezichtshoek."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from simulatie.sensoren.sensor import BaseSensor, SensorReading
from simulatie.config import DroneConfig

if TYPE_CHECKING:
    from simulatie.world import World
    from simulatie.entities.drone import Drone


class ProximitySensor(BaseSensor):
    """
    Detecteert alle entiteiten binnen sensor_range en sensor_fov.

    Confidence daalt lineair met afstand: conf = 1 - (dist / range).
    """

    def __init__(self, config: DroneConfig | None = None) -> None:
        cfg = config or DroneConfig()
        self.sensor_range: float = cfg.sensor_range
        self.sensor_fov: float = cfg.sensor_fov  # rad, 2π = 360°

    def scan(self, drone: Drone, world: World) -> list[SensorReading]:
        """Geeft alle gedetecteerde entiteiten terug als SensorReading-lijst."""
        readings: list[SensorReading] = []
        candidates = world.query_radius(drone.position, self.sensor_range)

        for entity in candidates:
            if entity.id == drone.id:
                continue

            delta = entity.position - drone.position
            dist = float(np.linalg.norm(delta))

            if dist > self.sensor_range:
                continue

            # Gezichtshoekcheck: sla over als buiten FOV
            if self.sensor_fov < 2 * math.pi - 1e-6:
                angle_to = math.atan2(float(delta[1]), float(delta[0]))
                angle_diff = abs(
                    math.atan2(
                        math.sin(angle_to - drone.heading),
                        math.cos(angle_to - drone.heading),
                    )
                )
                if angle_diff > self.sensor_fov / 2:
                    continue

            # Lineaire confidence: 1.0 dichtbij, 0.0 op de grens
            confidence = max(0.0, 1.0 - dist / self.sensor_range)

            readings.append(SensorReading(
                drone_id=drone.id,
                entity_id=entity.id,
                absolute_position=entity.position.copy(),
                confidence=confidence,
            ))

        return readings
