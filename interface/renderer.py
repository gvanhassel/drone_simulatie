"""Pygame hoofdrenderer: composeert heatmap, entiteiten en HUD."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simulatie.world import World
    from piloot.interface import OperatorState
    from interface.camera import Camera


class Renderer:
    """
    Composeert alle visuele lagen op het pygame-venster.

    Laagvolgorde:
        1. Heatmap (of kale wereld als heatmap uitstaat)
        2. Drone- en NPC-markers
        3. HUD (rechter panel)
    """

    def __init__(self, screen_w: int, screen_h: int, heatmap_ratio: float = 0.70) -> None:
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.heatmap_w = int(screen_w * heatmap_ratio)
        self.hud_x = self.heatmap_w
        self.hud_w = screen_w - self.heatmap_w
        self._screen = None
        self._heatmap_surface = None

        from interface.camera import Camera
        from interface.heatmap import HeatmapRenderer
        from interface.hud import HUD

        self.camera = Camera(self.heatmap_w, screen_h)
        self.heatmap_renderer = HeatmapRenderer(self.heatmap_w, screen_h)
        self.hud = HUD(self.hud_x, self.hud_w, screen_h)

    def init(self) -> None:
        """Initialiseer pygame en maak het venster aan."""
        import pygame
        pygame.init()
        pygame.display.set_caption("Drone Zwerm Simulatie")
        self._screen = pygame.display.set_mode((self.screen_w, self.screen_h))
        self._heatmap_surface = pygame.Surface((self.heatmap_w, self.screen_h))

    def draw_frame(
        self,
        world,
        fusion_data: np.ndarray,
        operator_state,
        fps: float = 0.0,
    ) -> None:
        """Render één volledig frame."""
        import pygame

        drones = world.get_drones()
        npcs = world.get_npcs()

        # ── Heatmap panel ──
        if operator_state.show_heatmap:
            self.heatmap_renderer.render(
                self._heatmap_surface,
                fusion_data,
                drones,
                operator_state.waypoints if hasattr(operator_state, 'waypoints') else [],
                self.camera,
                operator_state.selected_drone_id,
            )
        else:
            # Kale wereld: donkere achtergrond + NPC stippen
            self._heatmap_surface.fill((15, 15, 25))
            self._draw_npcs(self._heatmap_surface, npcs)
            for drone in drones:
                self.heatmap_renderer._draw_drone_marker(
                    self._heatmap_surface, drone, self.camera,
                    drone.id == operator_state.selected_drone_id
                )

        self._screen.blit(self._heatmap_surface, (0, 0))

        # ── HUD panel ──
        waypoint_count = len(operator_state.waypoints) if hasattr(operator_state, 'waypoints') else 0
        self.hud.render(self._screen, operator_state, drones, waypoint_count, fps)

        pygame.display.flip()

    def _draw_npcs(self, surface, npcs: list) -> None:
        """Teken NPC-stippen als kleine rode cirkels."""
        import pygame
        for npc in npcs:
            sp = self.camera.world_to_screen(npc.position)
            sx, sy = int(sp[0]), int(sp[1])
            if 0 <= sx < self.heatmap_w and 0 <= sy < self.screen_h:
                pygame.draw.circle(surface, (220, 80, 80), (sx, sy), 4)

    def quit(self) -> None:
        import pygame
        pygame.quit()
