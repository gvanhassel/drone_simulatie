"""Tests voor de simulatiewereld, spatiale grid en SimClock."""

import time
import numpy as np
import pytest

from simulatie.config import SimConfig
from simulatie.clock import SimClock
from simulatie.world import World, SpatialGrid
from simulatie.entities.base_entity import BaseEntity


# ── Hulpklasse voor testen ──────────────────────────────────────────────────

class DummyEntity(BaseEntity):
    """Minimale concrete implementatie van BaseEntity voor tests."""

    def __init__(self, position, radius=10.0):
        super().__init__(position, radius)
        self.update_calls = 0

    def update(self, dt, world):
        self.update_calls += 1


# ── SpatialGrid tests ───────────────────────────────────────────────────────

class TestSpatialGrid:

    def test_insert_en_query_radius(self):
        grid = SpatialGrid(1000, 1000, 100)
        e = DummyEntity([500, 500])
        grid.insert(e)
        ids = grid.get_ids_in_cells(grid.get_nearby_cells(np.array([500, 500]), 50))
        assert e.id in ids

    def test_remove_verwijdert_entiteit(self):
        grid = SpatialGrid(1000, 1000, 100)
        e = DummyEntity([200, 200])
        grid.insert(e)
        grid.remove(e.id)
        ids = grid.get_ids_in_cells(grid.get_nearby_cells(np.array([200, 200]), 50))
        assert e.id not in ids

    def test_update_herplaatst_entiteit(self):
        grid = SpatialGrid(1000, 1000, 100)
        e = DummyEntity([50, 50])
        grid.insert(e)
        e.position = np.array([950, 950], dtype=np.float32)
        grid.update(e)
        ids_oud = grid.get_ids_in_cells(grid.get_nearby_cells(np.array([50, 50]), 50))
        ids_nieuw = grid.get_ids_in_cells(grid.get_nearby_cells(np.array([950, 950]), 50))
        assert e.id not in ids_oud
        assert e.id in ids_nieuw

    def test_remove_niet_bestaande_entiteit_geen_fout(self):
        grid = SpatialGrid(1000, 1000, 100)
        import uuid
        grid.remove(uuid.uuid4())  # Mag geen exception gooien


# ── World tests ──────────────────────────────────────────────────────────────

class TestWorld:

    @pytest.fixture
    def world(self):
        return World(SimConfig())

    def test_add_en_query_radius(self, world):
        e = DummyEntity([1000, 1000])
        world.add_entity(e)
        resultaat = world.query_radius(np.array([1000, 1000]), 50)
        assert e in resultaat

    def test_query_radius_buiten_bereik(self, world):
        e = DummyEntity([100, 100])
        world.add_entity(e)
        resultaat = world.query_radius(np.array([900, 900]), 50)
        assert e not in resultaat

    def test_remove_entity(self, world):
        e = DummyEntity([500, 500])
        world.add_entity(e)
        world.remove_entity(e.id)
        assert world.get_entity(e.id) is None
        assert world.entity_count == 0

    def test_step_roept_update_aan(self, world):
        e = DummyEntity([500, 500])
        world.add_entity(e)
        world.step(0.016)
        assert e.update_calls == 1

    def test_query_radius_exacte_grens(self, world):
        e = DummyEntity([200, 200])
        world.add_entity(e)
        # Precies op de grens (100px afstand, radius=100)
        resultaat = world.query_radius(np.array([300, 200]), 100.0)
        assert e in resultaat

    def test_meerdere_entiteiten_in_radius(self, world):
        entiteiten = [DummyEntity([500 + i * 10, 500]) for i in range(5)]
        for e in entiteiten:
            world.add_entity(e)
        resultaat = world.query_radius(np.array([500, 500]), 60)
        assert len(resultaat) >= 5

    def test_clamp_position(self, world):
        pos = np.array([-100, 3000], dtype=np.float32)
        geclampt = world.clamp_position(pos)
        assert geclampt[0] == 0
        assert geclampt[1] == world.height


# ── BaseEntity tests ─────────────────────────────────────────────────────────

class TestBaseEntity:

    def test_unieke_ids(self):
        a = DummyEntity([0, 0])
        b = DummyEntity([0, 0])
        assert a.id != b.id

    def test_distance_to(self):
        a = DummyEntity([0, 0])
        b = DummyEntity([3, 4])
        assert abs(a.distance_to(b) - 5.0) < 1e-5

    def test_to_dict_bevat_velden(self):
        e = DummyEntity([100, 200])
        d = e.to_dict()
        assert "id" in d
        assert "position" in d
        assert "velocity" in d
        assert d["position"] == [100.0, 200.0]


# ── SimClock tests ────────────────────────────────────────────────────────────

class TestSimClock:

    def test_tick_geeft_positieve_dt(self):
        klok = SimClock(60)
        time.sleep(0.01)
        dt = klok.tick()
        assert dt > 0

    def test_dt_begrensd_op_drievoud_target(self):
        klok = SimClock(60)
        time.sleep(0.2)  # Simuleer grote lag
        dt = klok.tick()
        assert dt <= (1.0 / 60) * 3 + 1e-6

    def test_elapsed_neemt_toe(self):
        klok = SimClock(60)
        time.sleep(0.02)
        klok.tick()
        assert klok.elapsed > 0

    def test_fps_berekening(self):
        klok = SimClock(60)
        for _ in range(10):
            time.sleep(1 / 60)
            klok.tick()
        assert klok.fps > 0
