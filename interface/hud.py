"""HUD: toont statistieken, drone-telemetrie en formatie-knoppen."""

from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from piloot.interface import OperatorState
    from simulatie.entities.drone import Drone


class HUD:
    """Rendert het informatiegedeelte rechts van de heatmap."""

    BG_COLOR = (20, 20, 35)
    TEXT_COLOR = (220, 220, 220)
    ACCENT_COLOR = (80, 200, 120)
    WARN_COLOR = (255, 160, 50)
    PANEL_PADDING = 12

    def __init__(self, panel_x: int, panel_w: int, panel_h: int) -> None:
        self.panel_x = panel_x
        self.panel_w = panel_w
        self.panel_h = panel_h
        self._fps_history: deque[float] = deque(maxlen=60)
        self._font_small = None
        self._font_medium = None
        self._font_large = None

    def _ensure_fonts(self) -> None:
        if self._font_small is None:
            import pygame
            pygame.font.init()
            # Gebruik pygame's ingebouwde font (geen fc-list/systeemfonts nodig)
            self._font_small = pygame.font.Font(None, 16)
            self._font_medium = pygame.font.Font(None, 19)
            self._font_large = pygame.font.Font(None, 22)

    def update_fps(self, fps: float) -> None:
        self._fps_history.append(fps)

    def render(
        self,
        surface,
        operator_state: OperatorState,
        drones: list[Drone],
        waypoint_count: int,
        fps: float,
    ) -> None:
        """Render het HUD-panel."""
        import pygame

        self._ensure_fonts()
        self.update_fps(fps)
        avg_fps = sum(self._fps_history) / max(len(self._fps_history), 1)

        # Achtergrond
        panel_rect = pygame.Rect(self.panel_x, 0, self.panel_w, self.panel_h)
        pygame.draw.rect(surface, self.BG_COLOR, panel_rect)
        pygame.draw.line(surface, (50, 50, 80), (self.panel_x, 0), (self.panel_x, self.panel_h), 2)

        y = self.PANEL_PADDING
        x = self.panel_x + self.PANEL_PADDING

        def draw(text, font=None, color=None):
            nonlocal y
            f = font or self._font_medium
            c = color or self.TEXT_COLOR
            surf = f.render(text, True, c)
            surface.blit(surf, (x, y))
            y += surf.get_height() + 4

        def divider():
            nonlocal y
            pygame.draw.line(surface, (50, 50, 80),
                             (self.panel_x + 5, y + 2),
                             (self.panel_x + self.panel_w - 5, y + 2), 1)
            y += 8

        # ── Titre ──
        draw("ZWERM STATUS", self._font_large, self.ACCENT_COLOR)
        divider()

        # ── Statistieken ──
        state_label = "|| GEPAUZEERD" if operator_state.paused else "> ACTIEF"
        draw(state_label, color=self.WARN_COLOR if operator_state.paused else self.ACCENT_COLOR)
        draw(f"Drones:    {len(drones)}")
        draw(f"Waypoints: {waypoint_count}")
        draw(f"FPS:       {avg_fps:.0f}")
        divider()

        # ── Geselecteerde drone ──
        draw("GESELECTEERDE DRONE", self._font_large, self.ACCENT_COLOR)
        divider()
        selected = next((d for d in drones if d.id == operator_state.selected_drone_id), None)
        if selected:
            import numpy as np
            speed = float(np.linalg.norm(selected.velocity))
            draw(f"ID:    ...{str(selected.id)[-6:]}")
            draw(f"Pos:   ({selected.position[0]:.0f}, {selected.position[1]:.0f})")
            draw(f"Speed: {speed:.0f} px/s")
            draw(f"State: {selected.state.value.upper()}")
        else:
            draw("(klik op drone)")
        divider()

        # ── Formatie ──
        draw("FORMATIE", self._font_large, self.ACCENT_COLOR)
        divider()
        formation_names = {0: "[0] Vrij (flocking)", 1: "[1] V-formatie",
                           2: "[2] Cirkel", 3: "[3] Grid"}
        for mode, name in formation_names.items():
            color = self.ACCENT_COLOR if mode == operator_state.formation_mode else self.TEXT_COLOR
            draw(name, color=color)
        divider()

        # ── Sneltoetsen ──
        draw("SNELTOETSEN", self._font_large, self.ACCENT_COLOR)
        divider()
        draw("SPACE  pauze/hervat", self._font_small)
        draw("H      heatmap toggle", self._font_small)
        draw("S      sensor radii", self._font_small)
        draw("R      reset heatmap", self._font_small)
        draw("ESC    afsluiten", self._font_small)
