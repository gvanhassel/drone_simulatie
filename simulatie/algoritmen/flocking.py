"""
Reynolds flocking algoritme (Vasarhelyi-variant) voor drone-zwermcoördinatie.

Volledig vectorized met numpy broadcasting — O(N²) maar zonder Python for-loops
over drone-paren. Gebaseerd op SwarmLab (Soria et al., 2020) en Marek et al. (2024).

Drie klassieke krachten:
    - Separatie:  vermijd botsingen met te-dichtbij drones
    - Uitlijning: match snelheid met buurzwerm
    - Coherentie: beweeg richting massacentrum van buren

Plus waypoint-attractie als operator een waypoint heeft ingesteld.
"""

from __future__ import annotations

from uuid import UUID

import numpy as np

from simulatie.config import FlockingConfig, DroneConfig


class FlockingController:
    """
    Berekent Reynolds-stuurkrachten voor alle drones tegelijk.

    Gebruik:
        forces = controller.compute_forces(drones)
        for drone in drones:
            drone.apply_force(forces[drone.id])
    """

    def __init__(
        self,
        config: FlockingConfig | None = None,
        drone_config: DroneConfig | None = None,
    ) -> None:
        self.config = config or FlockingConfig()
        self.drone_config = drone_config or DroneConfig()

    def compute_forces(self, drones: list) -> dict[UUID, np.ndarray]:
        """
        Berekent de gecombineerde stuurkracht per drone.

        Returns:
            Dict van drone-UUID naar krachtvector (np.ndarray shape (2,)).
        """
        n = len(drones)
        if n == 0:
            return {}
        if n == 1:
            # Alleen waypoint-attractie voor solitaire drone
            return {drones[0].id: self._single_waypoint_force(drones[0])}

        # Bouw matrix-representaties — shape (N, 2)
        positions = np.stack([d.position for d in drones], axis=0).astype(np.float64)
        velocities = np.stack([d.velocity for d in drones], axis=0).astype(np.float64)

        # Pairwise afstands-berekening — kern van het algoritme
        # diff[i,j] = position[i] - position[j]
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]  # (N, N, 2)
        dist = np.linalg.norm(diff, axis=2)  # (N, N)

        # Maskers
        sep_mask = (dist < self.config.separation_radius) & (dist > 1e-8)    # (N, N)
        nbr_mask = (dist < self.config.neighbor_radius) & (dist > 1e-8)      # (N, N)

        # Drie krachten
        sep_force = self._separation(diff, dist, sep_mask)        # (N, 2)
        aln_force = self._alignment(velocities, nbr_mask)          # (N, 2)
        coh_force = self._cohesion(positions, nbr_mask)             # (N, 2)
        way_force = self._waypoint_attraction(drones, n)            # (N, 2)

        total = sep_force + aln_force + coh_force + way_force      # (N, 2)

        # Clamp op max_acceleration
        max_accel = self.drone_config.max_acceleration
        magnitudes = np.linalg.norm(total, axis=1, keepdims=True)  # (N, 1)
        scale = np.where(magnitudes > max_accel, max_accel / (magnitudes + 1e-12), 1.0)
        total = total * scale

        return {drone.id: total[i].astype(np.float32) for i, drone in enumerate(drones)}

    # ── Separatie ─────────────────────────────────────────────────────────────

    def _separation(
        self,
        diff: np.ndarray,   # (N, N, 2) — diff[i,j] = pos[i] - pos[j]
        dist: np.ndarray,   # (N, N)
        mask: np.ndarray,   # (N, N) bool — True = te dichtbij
    ) -> np.ndarray:
        """
        Duw drones weg van buren die te dichtbij zijn.
        Kracht stijgt lineair naarmate buur dichterbij komt:
            magnitude = (sep_radius - dist) / sep_radius
        """
        sep_radius = self.config.separation_radius
        safe_dist = np.where(mask, dist, 1.0)                # voorkom /0
        # Eenheidsvector weg van buur
        normalized = diff / safe_dist[..., np.newaxis]       # (N, N, 2)
        # Lineair gewicht: sterker bij kleine afstand
        strength = np.where(mask, (sep_radius - dist) / sep_radius, 0.0)  # (N, N)
        weighted = normalized * strength[..., np.newaxis]    # (N, N, 2)
        force = weighted.sum(axis=1)                         # (N, 2)
        return force * self.config.separation_weight

    # ── Uitlijning ────────────────────────────────────────────────────────────

    def _alignment(
        self,
        velocities: np.ndarray,  # (N, 2)
        mask: np.ndarray,        # (N, N) bool
    ) -> np.ndarray:
        """Stuur richting gemiddelde snelheid van buren. Nul als geen buren."""
        count = mask.sum(axis=1, keepdims=True).astype(np.float64)  # (N, 1)
        has_neighbors = (count > 0)                                  # (N, 1) bool
        safe_count = np.where(has_neighbors, count, 1.0)
        avg_vel = (velocities[np.newaxis, :, :] * mask[..., np.newaxis]).sum(axis=1) / safe_count
        force = (avg_vel - velocities) * self.config.alignment_weight
        return force * has_neighbors  # nul als geen buren

    # ── Coherentie ────────────────────────────────────────────────────────────

    def _cohesion(
        self,
        positions: np.ndarray,  # (N, 2)
        mask: np.ndarray,       # (N, N) bool
    ) -> np.ndarray:
        """Stuur richting massacentrum van buren. Nul als geen buren."""
        count = mask.sum(axis=1, keepdims=True).astype(np.float64)  # (N, 1)
        has_neighbors = (count > 0)                                  # (N, 1) bool
        safe_count = np.where(has_neighbors, count, 1.0)
        center = (positions[np.newaxis, :, :] * mask[..., np.newaxis]).sum(axis=1) / safe_count
        force = (center - positions) * self.config.cohesion_weight
        return force * has_neighbors  # nul als geen buren

    # ── Waypoint-attractie ────────────────────────────────────────────────────

    def _waypoint_attraction(self, drones: list, n: int) -> np.ndarray:
        """Trek elke drone richting zijn toegewezen waypoint."""
        forces = np.zeros((n, 2), dtype=np.float64)
        for i, drone in enumerate(drones):
            if drone.target_position is not None:
                delta = drone.target_position.astype(np.float64) - drone.position.astype(np.float64)
                dist = np.linalg.norm(delta)
                if dist > drone.config.arrival_radius:
                    forces[i] = (delta / dist) * self.config.waypoint_weight
        return forces

    def _single_waypoint_force(self, drone) -> np.ndarray:
        """Waypoint-kracht voor een alleenstaande drone."""
        if drone.target_position is not None:
            delta = drone.target_position.astype(np.float64) - drone.position.astype(np.float64)
            dist = np.linalg.norm(delta)
            if dist > drone.config.arrival_radius:
                force = (delta / dist) * self.config.waypoint_weight
                max_accel = self.drone_config.max_acceleration
                mag = np.linalg.norm(force)
                if mag > max_accel:
                    force = force / mag * max_accel
                return force.astype(np.float32)
        return np.zeros(2, dtype=np.float32)
