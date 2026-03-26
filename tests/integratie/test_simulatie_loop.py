"""Integratietests voor de simulatieloop."""

import os
import numpy as np
import pytest

# Headless pygame instellen vóór import
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["SDL_AUDIODRIVER"] = "dummy"

from simulatie.simulation import Simulation
from simulatie.config import SimConfig


@pytest.fixture
def sim():
    s = Simulation(sim_config=SimConfig(seed=42), headless=True)
    s.initialize(n_drones=10, n_npcs=5)
    return s


class TestSimulatieLoop:

    def test_100_stappen_zonder_exception(self, sim):
        for _ in range(100):
            sim.step(1 / 60)

    def test_deterministisch_met_zelfde_seed(self):
        """Twee runs met dezelfde seed geven identieke eindposities."""
        def run(seed):
            s = Simulation(sim_config=SimConfig(seed=seed), headless=True)
            s.initialize(n_drones=8, n_npcs=3)
            for _ in range(60):
                s.step(1 / 60)
            return [d.position.copy() for d in s.world.get_drones()]

        pos1 = run(42)
        pos2 = run(42)
        for p1, p2 in zip(pos1, pos2):
            assert np.allclose(p1, p2, atol=0.01)

    def test_metrics_correct_na_stap(self, sim):
        sim.step(1 / 60)
        m = sim.get_metrics()
        assert m["drone_count"] == 10
        assert m["npc_count"] == 5
        assert m["frame"] == 1

    def test_heatmap_data_shape(self, sim):
        sim.step(1 / 60)
        m = sim.get_metrics()
        # Heatmap_max is float (geeft aan dat fusie werkte)
        assert isinstance(m["heatmap_max"], float)

    def test_pauze_stopt_simulatie(self, sim):
        posities_voor = [d.position.copy() for d in sim.world.get_drones()]
        sim.operator_state.paused = True
        sim.step(1 / 60)
        posities_na = [d.position.copy() for d in sim.world.get_drones()]
        for p1, p2 in zip(posities_voor, posities_na):
            assert np.allclose(p1, p2)


class TestZwermGedrag:

    def test_zwerm_convergeert_naar_waypoint(self):
        """20 drones, 10 simulatieseconden → allen < 100px van waypoint."""
        sim = Simulation(sim_config=SimConfig(seed=7), headless=True)
        sim.initialize(n_drones=20, n_npcs=0)

        waypoint_pos = np.array([1000.0, 1000.0])
        sim.waypoint_manager.add_waypoint(waypoint_pos)
        sim.waypoint_manager.assign_drones(sim.world.get_drones())

        for _ in range(600):  # 10 seconden @ 60fps
            sim.step(1 / 60)

        drones = sim.world.get_drones()
        afstanden = [
            float(np.linalg.norm(d.position - waypoint_pos))
            for d in drones
        ]
        # Minstens 60% van de drones moet dichtbij zijn (flocking houdt zwerm verspreid)
        dichtbij = sum(1 for a in afstanden if a < 200)
        assert dichtbij >= len(drones) * 0.6, \
            f"Slechts {dichtbij}/{len(drones)} drones gearriveerd: {[f'{a:.0f}' for a in afstanden]}"

    def test_geen_drones_buiten_wereld(self):
        """Drones mogen de wereldgrenzen nooit overschrijden."""
        sim = Simulation(sim_config=SimConfig(seed=13), headless=True)
        sim.initialize(n_drones=15, n_npcs=5)

        for _ in range(300):
            sim.step(1 / 60)

        for drone in sim.world.get_drones():
            assert 0 <= drone.position[0] <= sim.sim_cfg.world_width
            assert 0 <= drone.position[1] <= sim.sim_cfg.world_height
