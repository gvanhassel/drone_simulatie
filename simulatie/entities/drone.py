"""Drone-entiteit met kinematisch SITL-bewegingsmodel en state machine."""

from __future__ import annotations

import math
from enum import Enum
from typing import TYPE_CHECKING, Any
from uuid import UUID

import numpy as np

from simulatie.config import DroneConfig
from simulatie.entities.base_entity import BaseEntity

if TYPE_CHECKING:
    from simulatie.world import World


class DroneState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    HOVERING = "hovering"
    EMERGENCY = "emergency"


class Drone(BaseEntity):
    """
    Drone-entiteit met kinematisch bewegingsmodel.

    Kinematisch model (gebaseerd op Marek et al., 2024):
        desired = normalize(target - pos) * max_speed
        steering = clamp(desired - velocity, max_accel * dt)
        velocity = clamp(velocity + steering, max_speed)
        position += velocity * dt
    """

    def __init__(self, position: np.ndarray, config: DroneConfig | None = None) -> None:
        super().__init__(position, radius=config.collision_radius if config else 10.0)
        self.config = config or DroneConfig()
        self.state: DroneState = DroneState.IDLE
        self.target_position: np.ndarray | None = None
        self.assigned_waypoint_id: UUID | None = None

        # Wordt gevuld door ProximitySensor.scan()
        self.sensor_readings: list = []

        # Kracht opgelegd door FlockingController
        self._applied_force: np.ndarray = np.zeros(2, dtype=np.float32)

        # Teller voor emergency-cooldown
        self._emergency_timer: float = 0.0
        self._emergency_duration: float = 1.0  # seconden

    # ── Publieke API ──────────────────────────────────────────────────────────

    def set_target(self, pos: np.ndarray) -> None:
        """Stel een nieuw doelwaypoint in (wereldcoördinaten)."""
        self.target_position = np.array(pos, dtype=np.float32)
        if self.state not in (DroneState.EMERGENCY,):
            self.state = DroneState.MOVING

    def apply_force(self, force: np.ndarray) -> None:
        """Sla de flockkracht op — wordt verwerkt in update()."""
        self._applied_force = np.array(force, dtype=np.float32)

    def emergency_stop(self) -> None:
        """Activeer noodstop: drone remt volledig af."""
        self.state = DroneState.EMERGENCY
        self._emergency_timer = self._emergency_duration

    # ── Update loop ───────────────────────────────────────────────────────────

    def update(self, dt: float, world: World) -> None:
        """Werk drone bij: state machine → kinematica → grensbewaking."""
        self._update_state(dt, world)

        if self.state == DroneState.EMERGENCY:
            self._apply_braking(dt)
        elif self.state == DroneState.MOVING:
            self._apply_kinematic_model(dt)
        elif self.state == DroneState.HOVERING:
            self._apply_damping(dt)
        # IDLE: geen beweging

        # Houd de drone binnen de wereldgrenzen
        self.position = world.clamp_position(self.position)
        self._update_heading_from_velocity()

        # Reset toegepaste kracht voor volgende frame
        self._applied_force = np.zeros(2, dtype=np.float32)

    # ── Interne methoden ──────────────────────────────────────────────────────

    def _update_state(self, dt: float, world: World) -> None:
        """State machine transities."""
        if self.state == DroneState.EMERGENCY:
            self._emergency_timer -= dt
            if self._emergency_timer <= 0:
                # Controleer of het veilig is om te hervatten
                if not self._is_collision_imminent(world):
                    self.state = DroneState.HOVERING
            return

        # Controleer op naderend botsingsgevaar
        if self._is_collision_imminent(world):
            self.emergency_stop()
            return

        if self.state == DroneState.MOVING and self.target_position is not None:
            dist = self.distance_to_point(self.target_position)
            if dist < self.config.arrival_radius:
                self.state = DroneState.HOVERING
                self.target_position = None

    def _is_collision_imminent(self, world: World) -> bool:
        """Geeft True terug als een andere drone te dichtbij is."""
        threshold = self.config.collision_radius * 2.5
        neighbors = world.query_radius(self.position, threshold)
        for other in neighbors:
            if isinstance(other, Drone) and other.id != self.id:
                if self.distance_to(other) < self.config.collision_radius * 2:
                    return True
        return False

    def _apply_kinematic_model(self, dt: float) -> None:
        """Kinematisch stuurmodel met flocking-kracht integratie."""
        cfg = self.config

        # Waypoint-attractie
        waypoint_force = np.zeros(2, dtype=np.float32)
        if self.target_position is not None:
            delta = self.target_position - self.position
            dist = float(np.linalg.norm(delta))
            if dist > cfg.arrival_radius:
                waypoint_force = (delta / dist) * cfg.max_speed

        # Gecombineerde gewenste snelheid
        desired_velocity = waypoint_force + self._applied_force

        # Begrens de gewenste snelheid op max_speed
        desired_speed = float(np.linalg.norm(desired_velocity))
        if desired_speed > cfg.max_speed:
            desired_velocity = (desired_velocity / desired_speed) * cfg.max_speed

        # Steering: verschil tussen gewenst en huidig, geclampt op max acceleratie
        steering = desired_velocity - self.velocity
        steering_mag = float(np.linalg.norm(steering))
        max_delta = cfg.max_acceleration * dt
        if steering_mag > max_delta:
            steering = (steering / steering_mag) * max_delta

        # Pas snelheid toe
        self.velocity = self.velocity + steering.astype(np.float32)

        # Begrens snelheid nogmaals
        speed = float(np.linalg.norm(self.velocity))
        if speed > cfg.max_speed:
            self.velocity = (self.velocity / speed) * cfg.max_speed

        # Positie bijwerken
        self.position = self.position + self.velocity * dt

    def _apply_braking(self, dt: float) -> None:
        """Rem af tijdens noodstop."""
        decel = self.config.max_acceleration * 2 * dt
        speed = float(np.linalg.norm(self.velocity))
        if speed <= decel:
            self.velocity = np.zeros(2, dtype=np.float32)
        else:
            self.velocity = self.velocity * (1 - decel / speed)
        self.position = self.position + self.velocity * dt

    def _apply_damping(self, dt: float) -> None:
        """Verminder snelheid geleidelijk bij hovering."""
        damping = max(0.0, 1.0 - 5.0 * dt)
        self.velocity = self.velocity * damping
        self.position = self.position + self.velocity * dt

    # ── Serialisatie ──────────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        d = super().to_dict()
        d.update({
            "state": self.state.value,
            "target_position": self.target_position.tolist() if self.target_position is not None else None,
            "speed": float(np.linalg.norm(self.velocity)),
        })
        return d
