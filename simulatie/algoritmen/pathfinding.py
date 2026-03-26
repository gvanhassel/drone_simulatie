"""A* pathfinding op een 2D grid met octagonale heuristiek."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class GridNode:
    x: int
    y: int

    def neighbors(self, max_x: int, max_y: int) -> list[GridNode]:
        """Geeft alle 8 aangrenzende knopen terug (inclusief diagonalen)."""
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = self.x + dx, self.y + dy
                if 0 <= nx < max_x and 0 <= ny < max_y:
                    result.append(GridNode(nx, ny))
        return result


@dataclass(order=True)
class _PQEntry:
    priority: float
    node: GridNode = field(compare=False)


def _octagonal_heuristic(a: GridNode, b: GridNode) -> float:
    """
    Octagonale (Chebyshev) afstandsheuristiek.
    Toelaatbaar voor 8-richting grid.
    """
    dx = abs(a.x - b.x)
    dy = abs(a.y - b.y)
    return max(dx, dy) + (1.4142135 - 1) * min(dx, dy)


class AStarPathfinder:
    """
    A* pathfinding op een rechthoekig 2D grid.

    Coördinaten zijn gridcellen; gebruik world_to_grid() / grid_to_world()
    voor conversie met wereldcoördinaten.
    """

    def __init__(self, world_w: int, world_h: int, cell_size: int = 50) -> None:
        self.world_w = world_w
        self.world_h = world_h
        self.cell_size = cell_size
        self.grid_w = world_w // cell_size
        self.grid_h = world_h // cell_size
        self._obstacles: set[GridNode] = set()

    # ── Conversie ─────────────────────────────────────────────────────────────

    def world_to_grid(self, pos: np.ndarray) -> GridNode:
        """Wereldcoördinaten → gridcel."""
        cx = int(np.clip(pos[0], 0, self.world_w - 1)) // self.cell_size
        cy = int(np.clip(pos[1], 0, self.world_h - 1)) // self.cell_size
        return GridNode(
            min(cx, self.grid_w - 1),
            min(cy, self.grid_h - 1),
        )

    def grid_to_world(self, node: GridNode) -> np.ndarray:
        """Gridcel → middelpunt van de cel in wereldcoördinaten."""
        return np.array([
            node.x * self.cell_size + self.cell_size // 2,
            node.y * self.cell_size + self.cell_size // 2,
        ], dtype=np.float32)

    # ── Obstakels ─────────────────────────────────────────────────────────────

    def update_obstacles(self, obstacles: set[GridNode]) -> None:
        self._obstacles = obstacles

    def add_obstacle(self, node: GridNode) -> None:
        self._obstacles.add(node)

    def remove_obstacle(self, node: GridNode) -> None:
        self._obstacles.discard(node)

    # ── Pathfinding ───────────────────────────────────────────────────────────

    def find_path(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        extra_obstacles: Optional[set[GridNode]] = None,
    ) -> list[np.ndarray]:
        """
        Zoekt een pad van start naar goal in wereldcoördinaten.

        Returns:
            Lijst van wereldcoördinaten (np.ndarray) langs het pad.
            Lege lijst als geen pad bestaat.
        """
        start_node = self.world_to_grid(start)
        goal_node = self.world_to_grid(goal)

        # Als start == goal hoeven we niets te doen
        if start_node == goal_node:
            return [self.grid_to_world(goal_node)]

        all_obstacles = self._obstacles.copy()
        if extra_obstacles:
            all_obstacles.update(extra_obstacles)

        # A* algoritme
        open_heap: list[_PQEntry] = []
        heapq.heappush(open_heap, _PQEntry(0.0, start_node))

        came_from: dict[GridNode, GridNode] = {}
        g_score: dict[GridNode, float] = {start_node: 0.0}
        f_score: dict[GridNode, float] = {
            start_node: _octagonal_heuristic(start_node, goal_node)
        }
        visited: set[GridNode] = set()

        while open_heap:
            current = heapq.heappop(open_heap).node

            if current == goal_node:
                return self._reconstruct_path(came_from, current)

            if current in visited:
                continue
            visited.add(current)

            for neighbor in current.neighbors(self.grid_w, self.grid_h):
                if neighbor in all_obstacles:
                    continue
                if neighbor in visited:
                    continue

                # Diagonale beweging kost meer
                dx = abs(neighbor.x - current.x)
                dy = abs(neighbor.y - current.y)
                step_cost = 1.4142135 if (dx + dy == 2) else 1.0

                tentative_g = g_score[current] + step_cost
                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + _octagonal_heuristic(neighbor, goal_node)
                    f_score[neighbor] = f
                    heapq.heappush(open_heap, _PQEntry(f, neighbor))

        return []  # Geen pad gevonden

    def _reconstruct_path(
        self, came_from: dict[GridNode, GridNode], current: GridNode
    ) -> list[np.ndarray]:
        """Reconstrueer het pad van goal naar start en keer om."""
        path = [self.grid_to_world(current)]
        while current in came_from:
            current = came_from[current]
            path.append(self.grid_to_world(current))
        path.reverse()
        return path
