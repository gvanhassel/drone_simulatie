"""NPC-entiteit die autonoom beweegt via A* en vlucht voor drones."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from simulatie.entities.base_entity import BaseEntity
from simulatie.algoritmen.pathfinding import AStarPathfinder, GridNode

if TYPE_CHECKING:
    from simulatie.world import World


class NPC(BaseEntity):
    """
    Niet-spelerpersonage dat autonoom ronddwaalt via A* pathfinding.
    Vlucht weg van drones die te dichtbij komen.
    """

    FLEE_RADIUS: float = 150.0    # px — binnen deze afstand vluchten
    WANDER_INTERVAL: float = 4.0  # seconden — hoe vaak nieuw doel
    SPEED: float = 60.0           # px/s
    ARRIVE_RADIUS: float = 30.0   # px — nabij genoeg = doel bereikt

    def __init__(
        self,
        position: np.ndarray,
        world_w: int = 2000,
        world_h: int = 2000,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(position, radius=8.0)
        self._pathfinder = AStarPathfinder(world_w, world_h, cell_size=50)
        self._world_w = world_w
        self._world_h = world_h
        self._rng = rng or random.Random()

        self._path: list[np.ndarray] = []
        self._path_index: int = 0
        self._wander_timer: float = 0.0
        self._is_fleeing: bool = False

    # ── Update loop ───────────────────────────────────────────────────────────

    def update(self, dt: float, world: World) -> None:
        """Beweeg richting huidig puntdoel; vlucht bij naderende drones."""
        self._wander_timer -= dt

        # Controleer of er drones dichtbij zijn
        nearby = world.query_radius(self.position, self.FLEE_RADIUS)
        drone_neighbors = [
            e for e in nearby
            if e.__class__.__name__ == "Drone" and e.id != self.id
        ]

        if drone_neighbors:
            self._flee_from_drones(drone_neighbors, world)
        elif self._wander_timer <= 0 or not self._has_valid_path():
            self._pick_new_wander_goal(world)

        self._follow_path(dt, world)

    # ── Bewegingslogica ───────────────────────────────────────────────────────

    def _flee_from_drones(self, drones: list, world: World) -> None:
        """Bereken vluchtrichting en stel een nieuw vluchtwaypoint in."""
        self._is_fleeing = True
        self._wander_timer = self.WANDER_INTERVAL / 2

        # Som van wegvectoren van alle naburige drones
        flee_vec = np.zeros(2, dtype=np.float32)
        for drone in drones:
            diff = self.position - drone.position
            dist = float(np.linalg.norm(diff))
            if dist > 0:
                flee_vec += diff / (dist * dist)  # gewogen op afstand

        flee_mag = float(np.linalg.norm(flee_vec))
        if flee_mag < 1e-6:
            return

        flee_dir = flee_vec / flee_mag
        flee_target = self.position + flee_dir * 300.0
        flee_target = np.clip(flee_target, [25, 25], [self._world_w - 25, self._world_h - 25])

        path = self._pathfinder.find_path(self.position, flee_target)
        if path:
            self._path = path
            self._path_index = 0

    def _pick_new_wander_goal(self, world: World) -> None:
        """Kies een willekeurig ronddwaalwaypoint."""
        self._is_fleeing = False
        self._wander_timer = self.WANDER_INTERVAL

        goal = np.array([
            self._rng.uniform(50, self._world_w - 50),
            self._rng.uniform(50, self._world_h - 50),
        ], dtype=np.float32)

        path = self._pathfinder.find_path(self.position, goal)
        if path:
            self._path = path
            self._path_index = 0

    def _follow_path(self, dt: float, world: World) -> None:
        """Beweeg stap voor stap langs het huidige pad."""
        if not self._path or self._path_index >= len(self._path):
            self._decelerate(dt)
            return

        target = self._path[self._path_index]
        delta = target - self.position
        dist = float(np.linalg.norm(delta))

        if dist < self.ARRIVE_RADIUS:
            self._path_index += 1
            return

        # Beweeg richting huidig puntdoel
        direction = delta / dist
        self.velocity = (direction * self.SPEED).astype(np.float32)
        self.position = self.position + self.velocity * dt
        self._update_heading_from_velocity()

    def _decelerate(self, dt: float) -> None:
        """Rem af als er geen pad meer is."""
        damping = max(0.0, 1.0 - 8.0 * dt)
        self.velocity = self.velocity * damping
        self.position = self.position + self.velocity * dt

    def _has_valid_path(self) -> bool:
        return bool(self._path) and self._path_index < len(self._path)

    # ── Serialisatie ──────────────────────────────────────────────────────────

    def to_dict(self):
        d = super().to_dict()
        d.update({
            "is_fleeing": self._is_fleeing,
            "path_remaining": len(self._path) - self._path_index,
        })
        return d
