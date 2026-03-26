"""OperatorInterface: verwerkt operatorinput (muis + toetsenbord) via pygame."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

if TYPE_CHECKING:
    from simulatie.entities.drone import Drone
    from piloot.waypoints import WaypointManager


@dataclass
class OperatorState:
    """Huidige staat van de operatorinterface."""
    paused: bool = False
    selected_drone_id: UUID | None = None
    formation_mode: int = 0             # 0=vrij, 1=V, 2=cirkel, 3=grid
    show_heatmap: bool = True
    show_sensor_radii: bool = False
    heatmap_panel_w: int = 896          # 70% van 1280px
    heatmap_panel_h: int = 720


class OperatorInterface:
    """
    Verwerkt pygame-events en vertaalt ze naar simulatieacties.

    Keyboard:
        SPACE   → pauzeert/hervat
        H       → wisselt heatmap/kale weergave
        S       → wisselt sensor-radii visualisatie
        R       → reset heatmap-accumulator
        0-3     → fortatiemode wisselen
        ESC     → afsluiten

    Muis (op heatmappanel):
        Links-klik      → waypoint plaatsen
        Rechts-klik     → dichtstbijzijnde waypoint verwijderen
        Klik op drone   → drone selecteren
        Scroll-wheel    → camera zoom
        Middle-drag     → camera pan
    """

    def __init__(
        self,
        state: OperatorState | None = None,
        waypoint_manager: WaypointManager | None = None,
    ) -> None:
        self.state = state or OperatorState()
        self.waypoint_manager = waypoint_manager
        self._quit_requested: bool = False
        self._middle_drag_last: tuple[int, int] | None = None

    @property
    def quit_requested(self) -> bool:
        return self._quit_requested

    def process_events(
        self,
        events: list,           # lijst van pygame.event.Event
        camera,
        drones: list[Drone],
    ) -> None:
        """Verwerk alle pygame-events voor dit frame."""
        try:
            import pygame
        except ImportError:
            return  # Headless modus zonder pygame

        for event in events:
            if event.type == pygame.QUIT:
                self._quit_requested = True

            elif event.type == pygame.KEYDOWN:
                self._handle_keydown(event.key, pygame)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mousedown(event, camera, drones, pygame)

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2:
                    self._middle_drag_last = None

            elif event.type == pygame.MOUSEMOTION:
                self._handle_mousemotion(event, camera, pygame)

            elif event.type == pygame.MOUSEWHEEL:
                camera.zoom_by(event.y * 0.1)

    # ── Keyboard ──────────────────────────────────────────────────────────────

    def _handle_keydown(self, key, pygame) -> None:
        if key == pygame.K_SPACE:
            self.state.paused = not self.state.paused
        elif key == pygame.K_h:
            self.state.show_heatmap = not self.state.show_heatmap
        elif key == pygame.K_s:
            self.state.show_sensor_radii = not self.state.show_sensor_radii
        elif key == pygame.K_ESCAPE:
            self._quit_requested = True
        elif key in (pygame.K_0, pygame.K_KP0):
            self.state.formation_mode = 0
        elif key in (pygame.K_1, pygame.K_KP1):
            self.state.formation_mode = 1
        elif key in (pygame.K_2, pygame.K_KP2):
            self.state.formation_mode = 2
        elif key in (pygame.K_3, pygame.K_KP3):
            self.state.formation_mode = 3

    # ── Muis ──────────────────────────────────────────────────────────────────

    def _handle_mousedown(self, event, camera, drones, pygame) -> None:
        mx, my = event.pos

        # Alleen acties op het heatmappanel (links deel scherm)
        if mx > self.state.heatmap_panel_w:
            return

        if event.button == 2:
            self._middle_drag_last = event.pos
            return

        world_pos = camera.screen_to_world(np.array([mx, my], dtype=np.float32))

        if event.button == 1:
            # Controleer eerst of we een drone aanikken
            clicked_drone = self._find_drone_at(world_pos, drones)
            if clicked_drone:
                self.state.selected_drone_id = clicked_drone.id
            else:
                # Waypoint plaatsen
                if self.waypoint_manager:
                    self.waypoint_manager.add_waypoint(world_pos)
                    self.waypoint_manager.assign_drones(drones)

        elif event.button == 3:
            # Dichtstbijzijnde waypoint verwijderen
            if self.waypoint_manager:
                self.waypoint_manager.remove_nearest(world_pos)

    def _handle_mousemotion(self, event, camera, pygame) -> None:
        if self._middle_drag_last is not None:
            dx = event.pos[0] - self._middle_drag_last[0]
            dy = event.pos[1] - self._middle_drag_last[1]
            camera.pan(-dx, -dy)
            self._middle_drag_last = event.pos

    def _find_drone_at(self, world_pos: np.ndarray, drones: list, radius: float = 20.0):
        """Vind een drone op of nabij world_pos."""
        for drone in drones:
            if float(np.linalg.norm(drone.position - world_pos)) < radius:
                return drone
        return None

    # ── Headless API (voor tests zonder pygame) ───────────────────────────────

    def place_waypoint(self, world_pos: np.ndarray, drones: list) -> None:
        """Programmatisch waypoint plaatsen (voor tests en headless modus)."""
        if self.waypoint_manager:
            self.waypoint_manager.add_waypoint(world_pos)
            self.waypoint_manager.assign_drones(drones)

    def remove_waypoint_at(self, world_pos: np.ndarray) -> None:
        """Programmatisch waypoint verwijderen."""
        if self.waypoint_manager:
            self.waypoint_manager.remove_nearest(world_pos)
