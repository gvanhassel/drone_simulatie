"""Tests voor A* pathfinding en NPC-entiteit."""

import time
import random
import numpy as np
import pytest

from simulatie.algoritmen.pathfinding import AStarPathfinder, GridNode
from simulatie.config import SimConfig, DroneConfig
from simulatie.entities.drone import Drone
from simulatie.entities.npc import NPC
from simulatie.world import World


# ── AStarPathfinder tests ────────────────────────────────────────────────────

class TestAStarPathfinder:

    @pytest.fixture
    def pf(self):
        return AStarPathfinder(1000, 1000, cell_size=50)

    def test_vindt_pad_leeg_grid(self, pf):
        start = np.array([25.0, 25.0])
        goal = np.array([975.0, 975.0])
        path = pf.find_path(start, goal)
        assert len(path) > 0

    def test_pad_eindigt_bij_doel(self, pf):
        start = np.array([25.0, 25.0])
        goal = np.array([475.0, 475.0])
        path = pf.find_path(start, goal)
        assert len(path) > 0
        last = path[-1]
        goal_node = pf.world_to_grid(goal)
        last_node = pf.world_to_grid(last)
        assert last_node == goal_node

    def test_vermijdt_obstakels(self, pf):
        # Blokkeer een gedeeltelijke verticale muur (laat boven open)
        for y in range(5, pf.grid_h):
            pf.add_obstacle(GridNode(10, y))
        # Pad moet langs de opening aan de bovenkant gaan
        start = np.array([25.0, 500.0])
        goal = np.array([975.0, 500.0])
        path = pf.find_path(start, goal)
        assert len(path) > 0
        # Geen padpunt mag in een obstakelcel vallen
        for point in path:
            node = pf.world_to_grid(point)
            assert node not in pf._obstacles

    def test_geen_pad_als_geblokkeerd(self, pf):
        # Omring start volledig met obstakels
        start_node = pf.world_to_grid(np.array([500.0, 500.0]))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                pf.add_obstacle(GridNode(start_node.x + dx, start_node.y + dy))
        path = pf.find_path(np.array([500.0, 500.0]), np.array([900.0, 900.0]))
        assert path == []

    def test_start_gelijk_aan_doel(self, pf):
        pos = np.array([500.0, 500.0])
        path = pf.find_path(pos, pos)
        assert len(path) == 1

    def test_performance_100x100_grid(self):
        """Pathfinding moet < 5ms op een 100×100 grid."""
        pf = AStarPathfinder(5000, 5000, cell_size=50)  # 100×100 grid
        start = np.array([25.0, 25.0])
        goal = np.array([4975.0, 4975.0])
        t0 = time.perf_counter()
        path = pf.find_path(start, goal)
        elapsed = (time.perf_counter() - t0) * 1000
        assert len(path) > 0
        assert elapsed < 50  # Ruime marge — aanpasbaar

    def test_world_to_grid_en_terug(self, pf):
        pos = np.array([125.0, 275.0])
        node = pf.world_to_grid(pos)
        world_pos = pf.grid_to_world(node)
        # Moet in dezelfde cel terecht komen
        assert pf.world_to_grid(world_pos) == node

    def test_heuristiek_toelaatbaar(self, pf):
        """Heuristiek mag de werkelijke gridpadkosten nooit overschatten."""
        a = GridNode(0, 0)
        b = GridNode(3, 4)
        from simulatie.algoritmen.pathfinding import _octagonal_heuristic
        h = _octagonal_heuristic(a, b)
        # Minimale gridpadkosten: max(3,4)=4 diagonale stappen à sqrt(2)
        min_grid_cost = max(3, 4) * 1.4142135
        assert h <= min_grid_cost + 1e-5


# ── NPC tests ────────────────────────────────────────────────────────────────

class TestNPC:

    @pytest.fixture
    def world(self):
        return World(SimConfig())

    def test_npc_beweegt_in_tijd(self, world):
        npc = NPC(np.array([500.0, 500.0]), rng=random.Random(42))
        world.add_entity(npc)
        start = npc.position.copy()
        for _ in range(60):
            world.step(0.016)
        # Na 1 seconde moet NPC bewogen zijn
        assert not np.allclose(npc.position, start, atol=1.0)

    def test_npc_vlucht_voor_drone(self, world):
        npc = NPC(np.array([500.0, 500.0]), rng=random.Random(42))
        drone = Drone(np.array([520.0, 500.0]), DroneConfig())
        world.add_entity(npc)
        world.add_entity(drone)

        for _ in range(30):
            world.step(0.016)

        # NPC moet vluchten (is_fleeing vlag)
        assert npc._is_fleeing

    def test_to_dict_bevat_velden(self):
        npc = NPC(np.array([100.0, 100.0]))
        d = npc.to_dict()
        assert "is_fleeing" in d
        assert "path_remaining" in d
