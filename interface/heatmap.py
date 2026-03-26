"""
HeatmapRenderer: visualiseert de gecombineerde sensordata als kleurheatmap.

Gebruikt een hardcoded plasma colormap LUT (256×3 uint8) — geen matplotlib
dependency. Gebaseerd op Soorati et al. (2021): heatmaps zijn empirisch
superieur voor enkelvoudige operatorbesturing van drone-zwermen.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from interface.camera import Camera
    from piloot.waypoints import Waypoint
    from simulatie.entities.drone import Drone


# Plasma colormap: 256 RGB-waarden van donkerblauw → geel/wit
# Gegenereerd van matplotlib.cm.plasma maar hier hardcoded voor geen dependency
_PLASMA_DATA = [
    (13,8,135),(16,7,136),(19,7,137),(22,7,138),(25,6,140),(27,6,141),(29,6,142),
    (32,5,143),(34,5,144),(37,5,145),(39,5,146),(41,5,147),(44,4,148),(46,4,149),
    (49,4,150),(51,4,151),(53,4,152),(56,3,153),(58,3,154),(60,3,155),(63,3,156),
    (65,3,157),(68,2,158),(70,2,159),(73,2,160),(75,2,161),(77,2,162),(80,2,162),
    (82,2,163),(84,1,164),(87,1,165),(89,1,166),(92,1,167),(94,1,167),(97,0,168),
    (99,0,169),(101,0,170),(104,0,170),(106,0,171),(109,0,172),(111,0,172),(113,0,173),
    (116,0,174),(118,0,174),(121,0,175),(123,0,175),(125,0,176),(128,1,177),(130,1,177),
    (133,1,178),(135,1,178),(137,2,178),(140,2,179),(142,3,179),(145,4,180),(147,4,180),
    (149,5,181),(152,6,181),(154,7,181),(157,8,182),(159,9,182),(161,10,182),(164,11,182),
    (166,13,183),(168,14,183),(171,15,183),(173,17,183),(176,18,183),(178,20,184),
    (180,21,184),(183,23,184),(185,24,184),(188,26,184),(190,28,184),(192,29,184),
    (195,31,184),(197,33,184),(199,35,184),(202,37,184),(204,39,184),(206,41,183),
    (209,43,183),(211,45,183),(213,47,182),(216,49,182),(218,52,181),(220,54,181),
    (222,56,180),(225,58,180),(227,61,179),(229,63,178),(231,65,178),(234,68,177),
    (236,70,176),(238,73,175),(240,75,174),(242,77,173),(244,80,172),(246,83,171),
    (248,85,170),(249,88,169),(251,90,168),(252,93,166),(253,96,165),(254,99,163),
    (254,102,162),(255,104,160),(255,107,158),(255,110,157),(255,113,155),(255,116,153),
    (255,119,151),(255,122,149),(255,125,147),(255,128,145),(255,131,143),(255,134,141),
    (255,137,139),(255,140,137),(255,143,135),(255,146,133),(255,149,130),(255,152,128),
    (255,155,126),(254,158,124),(254,161,121),(254,164,119),(254,167,117),(254,170,114),
    (253,173,112),(253,176,110),(253,179,107),(253,182,105),(252,185,102),(252,188,100),
    (252,191,97),(251,194,95),(251,197,92),(251,200,90),(250,203,87),(250,206,85),
    (249,209,82),(249,212,80),(248,215,77),(248,218,74),(248,221,72),(247,224,69),
    (247,227,66),(247,230,64),(246,233,61),(246,236,58),(245,239,56),(245,242,53),
    (244,245,50),(244,248,47),(243,251,45),(243,254,42),
]

# Vul op naar 256 als nodig
while len(_PLASMA_DATA) < 256:
    _PLASMA_DATA.append(_PLASMA_DATA[-1])


def _build_plasma_lut() -> np.ndarray:
    """Bouw een 256×3 uint8 LUT voor de plasma colormap."""
    lut = np.array(_PLASMA_DATA[:256], dtype=np.uint8)
    return lut


_PLASMA_LUT: np.ndarray = _build_plasma_lut()


class HeatmapRenderer:
    """
    Rendert de heatmap van sensor-fusie op een pygame Surface.

    Layout (70% van het scherm):
        - Kleurheatmap als achtergrond (donkerblauw=onbekend, geel/wit=actief)
        - Drone-markers: witte driehoek met heading
        - Waypoint-markers: gele cirkel met stippellijn naar toegewezen drones
    """

    DRONE_MARKER_SIZE: int = 10
    WAYPOINT_RADIUS: int = 12
    DRONE_COLOR = (255, 255, 255)
    WAYPOINT_COLOR = (255, 220, 0)
    SELECTED_COLOR = (0, 255, 128)

    def __init__(self, panel_w: int, panel_h: int) -> None:
        self.panel_w = panel_w
        self.panel_h = panel_h

    # ── Publieke render ───────────────────────────────────────────────────────

    def render(
        self,
        surface,                        # pygame.Surface
        fusion_data: np.ndarray,        # (grid_h, grid_w) float [0,1]
        drones: list[Drone],
        waypoints: list[Waypoint],
        camera: Camera,
        selected_drone_id=None,
    ) -> None:
        """Render het volledige heatmappanel op de gegeven surface."""
        import pygame

        # Heatmap als achtergrond
        self._draw_heatmap(surface, fusion_data)

        # Waypoint-markers + verbindingslijnen
        drone_by_id = {d.id: d for d in drones}
        for wp in waypoints:
            self._draw_waypoint(surface, wp, drone_by_id, camera)

        # Drone-markers
        for drone in drones:
            selected = (drone.id == selected_drone_id)
            self._draw_drone_marker(surface, drone, camera, selected)

    # ── Heatmap ───────────────────────────────────────────────────────────────

    def _draw_heatmap(self, surface, fusion_data: np.ndarray) -> None:
        """Zet fusion_data om naar RGB via plasma LUT en blit naar surface."""
        import pygame

        # Schaal naar panelresolutie
        grid_h, grid_w = fusion_data.shape
        # Indices voor LUT (0–255)
        indices = (fusion_data * 255).clip(0, 255).astype(np.uint8)  # (H, W)

        # RGB via LUT: (H, W) → (H, W, 3)
        rgb = _PLASMA_LUT[indices]  # numpy fancy indexing

        # Schaal naar panel grootte
        if grid_w != self.panel_w or grid_h != self.panel_h:
            # Eenvoudige nearest-neighbor upscale via herhalende indexen
            y_idx = (np.arange(self.panel_h) * grid_h / self.panel_h).astype(int)
            x_idx = (np.arange(self.panel_w) * grid_w / self.panel_w).astype(int)
            rgb = rgb[np.ix_(y_idx, x_idx)]  # (panel_h, panel_w, 3)

        # pygame verwacht (W, H, 3) voor surfarray
        rgb_t = rgb.transpose(1, 0, 2)
        pygame.surfarray.blit_array(surface, rgb_t)

    # ── Drone-marker ──────────────────────────────────────────────────────────

    def _draw_drone_marker(
        self, surface, drone: Drone, camera: Camera, selected: bool = False
    ) -> None:
        """Teken een driehoek die de heading van de drone aangeeft."""
        import pygame

        screen_pos = camera.world_to_screen(drone.position)
        sx, sy = int(screen_pos[0]), int(screen_pos[1])

        if not (0 <= sx < self.panel_w and 0 <= sy < self.panel_h):
            return  # Buiten zichtvenster

        color = self.SELECTED_COLOR if selected else self.DRONE_COLOR
        size = self.DRONE_MARKER_SIZE

        # Driehoek met tip in heading-richting
        angle = drone.heading
        tip = (sx + int(size * math.cos(angle)), sy + int(size * math.sin(angle)))
        left = (
            sx + int(size * 0.6 * math.cos(angle + 2.4)),
            sy + int(size * 0.6 * math.sin(angle + 2.4)),
        )
        right = (
            sx + int(size * 0.6 * math.cos(angle - 2.4)),
            sy + int(size * 0.6 * math.sin(angle - 2.4)),
        )
        pygame.draw.polygon(surface, color, [tip, left, right])

    # ── Waypoint-marker ───────────────────────────────────────────────────────

    def _draw_waypoint(
        self, surface, waypoint: Waypoint, drone_by_id: dict, camera: Camera
    ) -> None:
        """Teken een gele cirkel + stippellijnen naar toegewezen drones."""
        import pygame

        screen_pos = camera.world_to_screen(waypoint.position)
        sx, sy = int(screen_pos[0]), int(screen_pos[1])

        if not (0 <= sx < self.panel_w and 0 <= sy < self.panel_h):
            return

        pygame.draw.circle(surface, self.WAYPOINT_COLOR, (sx, sy), self.WAYPOINT_RADIUS, 2)

        # Stippellijn naar elke toegewezen drone
        for drone_id in waypoint.assigned_drone_ids:
            drone = drone_by_id.get(drone_id)
            if drone is None:
                continue
            ds = camera.world_to_screen(drone.position)
            dx, dy = int(ds[0]), int(ds[1])
            self._draw_dashed_line(surface, self.WAYPOINT_COLOR, (sx, sy), (dx, dy))

    @staticmethod
    def _draw_dashed_line(surface, color, start, end, dash_len: int = 8) -> None:
        """Teken een stippellijn van start naar end."""
        import pygame

        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < 1:
            return
        steps = int(length / dash_len)
        for i in range(0, steps, 2):
            t0 = i / steps
            t1 = min((i + 1) / steps, 1.0)
            p0 = (int(start[0] + dx * t0), int(start[1] + dy * t0))
            p1 = (int(start[0] + dx * t1), int(start[1] + dy * t1))
            pygame.draw.line(surface, color, p0, p1, 1)

    # ── Hulp: colormap ────────────────────────────────────────────────────────

    @staticmethod
    def get_lut() -> np.ndarray:
        """Geeft de plasma LUT terug (256×3 uint8)."""
        return _PLASMA_LUT.copy()
