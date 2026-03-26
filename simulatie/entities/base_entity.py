"""Abstracte basisklasse voor alle entiteiten in de simulatie."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

import numpy as np

if TYPE_CHECKING:
    from simulatie.world import World


class BaseEntity(ABC):
    """Basisklasse voor drone, NPC en andere gesimuleerde objecten."""

    def __init__(self, position: np.ndarray, radius: float = 10.0) -> None:
        self.id: UUID = uuid4()
        self.position: np.ndarray = np.array(position, dtype=np.float32)
        self.velocity: np.ndarray = np.zeros(2, dtype=np.float32)
        self.heading: float = 0.0  # radialen, 0 = rechts
        self.radius: float = radius

    @abstractmethod
    def update(self, dt: float, world: World) -> None:
        """Werk de entiteit bij voor één tijdstap dt (seconden)."""

    def distance_to(self, other: BaseEntity) -> float:
        """Euclidische afstand tot een andere entiteit (px)."""
        return float(np.linalg.norm(self.position - other.position))

    def distance_to_point(self, point: np.ndarray) -> float:
        """Euclidische afstand tot een punt (px)."""
        return float(np.linalg.norm(self.position - point))

    def to_dict(self) -> dict[str, Any]:
        """Serialiseer de entiteitstaat naar een dictionary."""
        return {
            "id": str(self.id),
            "type": self.__class__.__name__,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "heading": self.heading,
            "radius": self.radius,
        }

    def _update_heading_from_velocity(self) -> None:
        """Werk heading bij op basis van de huidige snelheidsvector."""
        speed = float(np.linalg.norm(self.velocity))
        if speed > 0.1:
            self.heading = math.atan2(float(self.velocity[1]), float(self.velocity[0]))
