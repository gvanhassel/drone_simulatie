"""Simulatiewereld met spatiale index voor efficiënte nabijheidsopvragen."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from simulatie.config import SimConfig
from simulatie.entities.base_entity import BaseEntity

if TYPE_CHECKING:
    pass


class SpatialGrid:
    """
    Cel-hash spatiale index voor O(1) insert/update/remove en
    efficiënte query_radius in O(k) waarbij k = entiteiten in buurt.
    """

    def __init__(self, world_w: int, world_h: int, cell_size: int) -> None:
        self.world_w = world_w
        self.world_h = world_h
        self.cell_size = cell_size
        self._cells: dict[tuple[int, int], set[UUID]] = {}
        self._entity_cells: dict[UUID, tuple[int, int]] = {}

    def _pos_to_cell(self, pos: np.ndarray) -> tuple[int, int]:
        """Zet wereldcoördinaten om naar een grid-cel."""
        cx = int(np.clip(pos[0], 0, self.world_w - 1)) // self.cell_size
        cy = int(np.clip(pos[1], 0, self.world_h - 1)) // self.cell_size
        return (cx, cy)

    def insert(self, entity: BaseEntity) -> None:
        """Voeg een entiteit toe aan de grid."""
        cell = self._pos_to_cell(entity.position)
        if cell not in self._cells:
            self._cells[cell] = set()
        self._cells[cell].add(entity.id)
        self._entity_cells[entity.id] = cell

    def remove(self, entity_id: UUID) -> None:
        """Verwijder een entiteit uit de grid."""
        if entity_id not in self._entity_cells:
            return
        cell = self._entity_cells.pop(entity_id)
        if cell in self._cells:
            self._cells[cell].discard(entity_id)
            if not self._cells[cell]:
                del self._cells[cell]

    def update(self, entity: BaseEntity) -> None:
        """Herplaats een entiteit na positiewijziging."""
        new_cell = self._pos_to_cell(entity.position)
        old_cell = self._entity_cells.get(entity.id)
        if new_cell == old_cell:
            return
        self.remove(entity.id)
        self.insert(entity)

    def get_nearby_cells(self, pos: np.ndarray, radius: float) -> list[tuple[int, int]]:
        """Geeft alle grid-cellen terug die de zoekradius overlappen."""
        cells = []
        min_cx = max(0, int(pos[0] - radius) // self.cell_size)
        max_cx = int(pos[0] + radius) // self.cell_size
        min_cy = max(0, int(pos[1] - radius) // self.cell_size)
        max_cy = int(pos[1] + radius) // self.cell_size
        for cx in range(min_cx, max_cx + 1):
            for cy in range(min_cy, max_cy + 1):
                cells.append((cx, cy))
        return cells

    def get_ids_in_cells(self, cells: list[tuple[int, int]]) -> set[UUID]:
        """Geeft alle entiteit-IDs terug in de opgegeven cellen."""
        result: set[UUID] = set()
        for cell in cells:
            if cell in self._cells:
                result.update(self._cells[cell])
        return result


class World:
    """
    Beheert alle entiteiten in de simulatiewereld en voert spatiale
    queries uit via SpatialGrid voor efficiënte nabijheidsdetectie.
    """

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.width = config.world_width
        self.height = config.world_height
        self._entities: dict[UUID, BaseEntity] = {}
        self._grid = SpatialGrid(
            config.world_width,
            config.world_height,
            config.spatial_cell_size,
        )

    def add_entity(self, entity: BaseEntity) -> None:
        """Voeg een entiteit toe aan de wereld."""
        self._entities[entity.id] = entity
        self._grid.insert(entity)

    def remove_entity(self, entity_id: UUID) -> None:
        """Verwijder een entiteit uit de wereld."""
        self._entities.pop(entity_id, None)
        self._grid.remove(entity_id)

    def get_entity(self, entity_id: UUID) -> BaseEntity | None:
        """Haal een entiteit op via UUID."""
        return self._entities.get(entity_id)

    def get_drones(self) -> list:
        """Geeft alle Drone-instanties terug."""
        from simulatie.entities.drone import Drone
        return [e for e in self._entities.values() if isinstance(e, Drone)]

    def get_npcs(self) -> list:
        """Geeft alle NPC-instanties terug."""
        from simulatie.entities.npc import NPC
        return [e for e in self._entities.values() if isinstance(e, NPC)]

    def query_radius(self, pos: np.ndarray, radius: float) -> list[BaseEntity]:
        """
        Geeft alle entiteiten terug binnen de gegeven radius van pos.
        Gebruikt spatiale grid voor efficiëntie.
        """
        nearby_cells = self._grid.get_nearby_cells(pos, radius)
        candidate_ids = self._grid.get_ids_in_cells(nearby_cells)
        radius_sq = radius * radius
        result = []
        for eid in candidate_ids:
            entity = self._entities.get(eid)
            if entity is None:
                continue
            diff = entity.position - pos
            if float(np.dot(diff, diff)) <= radius_sq:
                result.append(entity)
        return result

    def step(self, dt: float) -> None:
        """Werk alle entiteiten bij en herplaats ze in de spatiale grid."""
        for entity in list(self._entities.values()):
            entity.update(dt, self)
            self._grid.update(entity)

    def clamp_position(self, pos: np.ndarray) -> np.ndarray:
        """Begrens een positie tot de wereldgrenzen."""
        return np.clip(pos, [0, 0], [self.width, self.height]).astype(np.float32)

    @property
    def entity_count(self) -> int:
        return len(self._entities)
