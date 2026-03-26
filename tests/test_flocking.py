"""Tests voor het Reynolds flocking algoritme."""

import time
import numpy as np
import pytest

from simulatie.config import FlockingConfig, DroneConfig
from simulatie.entities.drone import Drone, DroneState
from simulatie.algoritmen.flocking import FlockingController


def maak_drones(posities: list, cfg: DroneConfig | None = None) -> list[Drone]:
    """Hulpfunctie: maak een lijst drones op opgegeven posities."""
    drone_cfg = cfg or DroneConfig()
    return [Drone(np.array(pos, dtype=np.float32), drone_cfg) for pos in posities]


@pytest.fixture
def controller():
    return FlockingController()


# ── Separatie tests ──────────────────────────────────────────────────────────

class TestSeparatie:

    def test_twee_drones_dicht_bij_elkaar_separatierichting_correct(self, controller):
        """Separatiekracht voor twee dichtbije drones wijst weg van buur."""
        drones = maak_drones([[500, 500], [510, 500]])  # 10px, < sep_radius=50
        # Bereken separatie direct (zonder coherentie/uitlijning)
        positions = np.array([[500.0, 500.0], [510.0, 500.0]])
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        sep_mask = (dist < controller.config.separation_radius) & (dist > 1e-8)
        sep = controller._separation(diff, dist, sep_mask)

        # Drone 0 links van drone 1: separatie moet negatief x geven (naar links)
        assert sep[0, 0] < 0
        # Drone 1 rechts van drone 0: separatie moet positief x geven (naar rechts)
        assert sep[1, 0] > 0

    def test_drones_ver_weg_geen_krachten(self, controller):
        """Drones buiten neighbor_radius hebben geen cohesie/uitlijningskracht."""
        # Plaatsen ver buiten neighbor_radius (150px)
        drones = maak_drones([[0, 0], [500, 500]])
        forces = controller.compute_forces(drones)
        # Zonder waypoint: krachten moeten nul zijn (geen buren in range)
        for force in forces.values():
            assert np.linalg.norm(force) < 1e-4


# ── Coherentie tests ─────────────────────────────────────────────────────────

class TestCoherentie:

    def test_loner_drone_wordt_richting_zwerm_getrokken(self, controller):
        """Een drone ver van de zwerm moet richting het centrum worden getrokken."""
        # Zwerm van 5 drones rondom (500, 500)
        zwerm_pos = [[490, 490], [510, 490], [490, 510], [510, 510], [500, 500]]
        loner_pos = [[200, 200]]  # ver van de zwerm, buiten neighbor_radius
        # Voeg loner toe vlakbij de zwerm (anders buiten neighbor_radius)
        drones = maak_drones([[490, 490], [510, 490], [490, 510], [510, 510], [500, 500], [450, 500]])
        forces = controller.compute_forces(drones)
        loner = drones[5]
        force = forces[loner.id]
        # Loner staat links (x=450), zwermcentrum rechts (x≈500) → kracht positief in x
        assert force[0] > 0


# ── Uitlijning tests ─────────────────────────────────────────────────────────

class TestUitlijning:

    def test_uitlijning_matcht_buursnelheid(self, controller):
        """Een stillstaande drone in een bewegende zwerm moet versnellen mee."""
        drones = maak_drones([[500, 500], [520, 500], [540, 500]])
        # Drones 1 en 2 bewegen naar rechts
        drones[1].velocity = np.array([50.0, 0.0], dtype=np.float32)
        drones[2].velocity = np.array([50.0, 0.0], dtype=np.float32)
        # Drone 0 staat stil

        forces = controller.compute_forces(drones)
        # Uitlijningskracht op drone 0 moet naar rechts wijzen
        assert forces[drones[0].id][0] > 0


# ── Waypoint-attractie tests ─────────────────────────────────────────────────

class TestWaypointAttractie:

    def test_drone_met_waypoint_krijgt_kracht_richting_doel(self, controller):
        drones = maak_drones([[500, 500]])
        drones[0].set_target(np.array([700.0, 500.0]))
        forces = controller.compute_forces(drones)
        assert forces[drones[0].id][0] > 0  # naar rechts


# ── Algemeene tests ───────────────────────────────────────────────────────────

class TestAlgemeen:

    def test_lege_dronelijst_geeft_leeg_dict(self, controller):
        assert controller.compute_forces([]) == {}

    def test_resulterende_kracht_begrensd_op_max_accel(self, controller):
        drones = maak_drones([[500 + i * 5, 500] for i in range(20)])
        forces = controller.compute_forces(drones)
        max_accel = DroneConfig().max_acceleration
        for force in forces.values():
            assert np.linalg.norm(force) <= max_accel + 1e-4

    def test_performance_50_drones(self, controller):
        """compute_forces moet < 10ms zijn voor 50 drones."""
        drones = maak_drones([[500 + i * 8, 500 + (i % 5) * 8] for i in range(50)])
        t0 = time.perf_counter()
        for _ in range(10):
            controller.compute_forces(drones)
        elapsed_per_call = (time.perf_counter() - t0) / 10 * 1000
        assert elapsed_per_call < 10, f"Te traag: {elapsed_per_call:.1f}ms"

    def test_geeft_kracht_voor_elke_drone(self, controller):
        drones = maak_drones([[500 + i * 20, 500] for i in range(8)])
        forces = controller.compute_forces(drones)
        assert len(forces) == len(drones)
        for drone in drones:
            assert drone.id in forces
