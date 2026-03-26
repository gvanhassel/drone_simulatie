"""Sensorabstracties en SensorReading datastructuur."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from simulatie.world import World
    from simulatie.entities.drone import Drone


class SensorReading(NamedTuple):
    """Enkelvoudige sensorwaarneming van een drone."""
    drone_id: UUID
    entity_id: UUID
    absolute_position: np.ndarray   # wereldcoördinaten van het gedetecteerde object
    confidence: float               # 0.0 – 1.0, daalt met afstand


class BaseSensor(ABC):
    """Abstracte basisklasse voor sensoren."""

    @abstractmethod
    def scan(self, drone: Drone, world: World) -> list[SensorReading]:
        """Scan de omgeving en geef een lijst van waarnemingen terug."""
