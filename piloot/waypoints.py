"""WaypointManager: beheert operatorwaypoints en wijst drones optimaal toe."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID, uuid4

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Waypoint:
    """Één operatorwaypoint in de simulatiewereld."""
    id: UUID = field(default_factory=uuid4)
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))
    assigned_drone_ids: list[UUID] = field(default_factory=list)
    arrival_radius: float = 20.0    # px — afstand waarop drone als gearriveerd geldt

    def is_drone_arrived(self, drone) -> bool:
        dist = float(np.linalg.norm(drone.position - self.position))
        return dist < self.arrival_radius


class WaypointManager:
    """
    Beheert waypoints en wijst drones optimaal toe via het Hongaarse algoritme.
    Gebaseerd op Hunt et al. (2023): mens-in-de-lus operatorbesturing.
    """

    def __init__(self) -> None:
        self._waypoints: dict[UUID, Waypoint] = {}

    # ── Publieke API ──────────────────────────────────────────────────────────

    def add_waypoint(self, world_pos: np.ndarray) -> Waypoint:
        """Voeg een nieuw waypoint toe en geef het terug."""
        wp = Waypoint(position=np.array(world_pos, dtype=np.float32))
        self._waypoints[wp.id] = wp
        return wp

    def remove_waypoint(self, waypoint_id: UUID) -> None:
        """Verwijder een waypoint. Wist ook drone-toewijzingen."""
        wp = self._waypoints.pop(waypoint_id, None)
        if wp:
            for drone_id in wp.assigned_drone_ids:
                # Drone-target wordt gereset door de simulatieloop
                pass

    def remove_nearest(self, world_pos: np.ndarray) -> Waypoint | None:
        """Verwijder het waypoint dichtstbij world_pos."""
        if not self._waypoints:
            return None
        nearest = min(
            self._waypoints.values(),
            key=lambda wp: float(np.linalg.norm(wp.position - world_pos)),
        )
        self.remove_waypoint(nearest.id)
        return nearest

    def assign_drones(self, drones: list) -> None:
        """
        Wijs drones optimaal toe aan waypoints.

        Meer drones dan waypoints: elke drone gaat naar het dichtstbijzijnde waypoint
        (meerdere drones per waypoint toegestaan).
        Meer waypoints dan drones: Hongaars algoritme voor optimale 1-op-1 toewijzing.
        """
        waypoints = list(self._waypoints.values())
        if not waypoints or not drones:
            return

        n_drones = len(drones)
        n_waypoints = len(waypoints)

        # Wis alle huidige toewijzingen
        for wp in waypoints:
            wp.assigned_drone_ids.clear()

        if n_drones <= n_waypoints:
            # Hongaars algoritme voor 1-op-1 toewijzing
            cost = np.zeros((n_drones, n_waypoints), dtype=np.float64)
            for i, drone in enumerate(drones):
                for j, wp in enumerate(waypoints):
                    cost[i, j] = float(np.linalg.norm(drone.position - wp.position))
            drone_idxs, wp_idxs = linear_sum_assignment(cost)
            for di, wj in zip(drone_idxs, wp_idxs):
                drone = drones[di]
                wp = waypoints[wj]
                wp.assigned_drone_ids.append(drone.id)
                drone.set_target(wp.position)
        else:
            # Meer drones dan waypoints: elk naar het dichtstbijzijnde waypoint
            wp_positions = np.array([wp.position for wp in waypoints], dtype=np.float64)
            for drone in drones:
                dists = np.linalg.norm(wp_positions - drone.position.astype(np.float64), axis=1)
                nearest_idx = int(np.argmin(dists))
                nearest_wp = waypoints[nearest_idx]
                nearest_wp.assigned_drone_ids.append(drone.id)
                drone.set_target(nearest_wp.position)

    def check_arrivals(self, drones: list) -> list[Waypoint]:
        """
        Verwijder waypoints waar alle toegewezen drones zijn gearriveerd.
        Geeft de verwijderde waypoints terug.
        """
        drone_by_id = {d.id: d for d in drones}
        to_remove = []

        for wp in list(self._waypoints.values()):
            if not wp.assigned_drone_ids:
                continue
            # Check of ALLE toegewezen drones zijn gearriveerd
            all_arrived = all(
                wp.is_drone_arrived(drone_by_id[did])
                for did in wp.assigned_drone_ids
                if did in drone_by_id
            )
            if all_arrived:
                to_remove.append(wp)

        for wp in to_remove:
            self.remove_waypoint(wp.id)

        return to_remove

    def get_all(self) -> list[Waypoint]:
        return list(self._waypoints.values())

    def clear(self) -> None:
        self._waypoints.clear()

    @property
    def count(self) -> int:
        return len(self._waypoints)
