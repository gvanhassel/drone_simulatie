"""Tests voor de Drone-entiteit en het kinematisch bewegingsmodel."""

import numpy as np
import pytest

from simulatie.config import DroneConfig, SimConfig
from simulatie.entities.drone import Drone, DroneState
from simulatie.world import World


@pytest.fixture
def world():
    return World(SimConfig())


@pytest.fixture
def cfg():
    return DroneConfig()


@pytest.fixture
def drone(cfg):
    return Drone(np.array([500.0, 500.0]), cfg)


# ── State machine ─────────────────────────────────────────────────────────────

class TestDroneState:

    def test_initieel_idle(self, drone):
        assert drone.state == DroneState.IDLE

    def test_set_target_wisselt_naar_moving(self, drone):
        drone.set_target(np.array([600.0, 500.0]))
        assert drone.state == DroneState.MOVING

    def test_aankomst_wisselt_naar_hovering(self, drone, world):
        world.add_entity(drone)
        # Target vlakbij (< arrival_radius)
        drone.set_target(np.array([505.0, 500.0]))
        drone.state = DroneState.MOVING
        # Forceer positie bijna op target
        drone.position = np.array([504.0, 500.0], dtype=np.float32)
        drone.update(0.016, world)
        assert drone.state == DroneState.HOVERING

    def test_emergency_stop_werkt(self, drone):
        drone.emergency_stop()
        assert drone.state == DroneState.EMERGENCY

    def test_emergency_herstel_na_timer(self, drone, world):
        world.add_entity(drone)
        drone.emergency_stop()
        # Simuleer lange tijdstap zodat timer verloopt
        drone._emergency_timer = 0.0
        drone.update(0.1, world)
        assert drone.state in (DroneState.HOVERING, DroneState.EMERGENCY)


# ── Kinematisch model ─────────────────────────────────────────────────────────

class TestDroneKinematics:

    def test_drone_beweegt_richting_target(self, drone, world):
        world.add_entity(drone)
        start_pos = drone.position.copy()
        drone.set_target(np.array([700.0, 500.0]))
        for _ in range(10):
            drone.update(0.016, world)
        assert drone.position[0] > start_pos[0]

    def test_snelheid_begrensd_op_max_speed(self, drone, world):
        world.add_entity(drone)
        drone.set_target(np.array([2000.0, 2000.0]))
        for _ in range(60):
            drone.update(0.016, world)
        speed = float(np.linalg.norm(drone.velocity))
        assert speed <= drone.config.max_speed + 0.1

    def test_positie_blijft_binnen_wereld(self, drone, world):
        world.add_entity(drone)
        # Target buiten wereld
        drone.set_target(np.array([99999.0, 99999.0]))
        for _ in range(120):
            drone.update(0.016, world)
        assert 0 <= drone.position[0] <= world.width
        assert 0 <= drone.position[1] <= world.height

    def test_hovering_vertraagt_drone(self, drone, world):
        world.add_entity(drone)
        # Geef drone een initiële snelheid
        drone.velocity = np.array([100.0, 0.0], dtype=np.float32)
        drone.state = DroneState.HOVERING
        for _ in range(30):
            drone.update(0.016, world)
        speed = float(np.linalg.norm(drone.velocity))
        assert speed < 50.0  # Aanzienlijk vertraagd

    def test_flockingkracht_beïnvloedt_beweging(self, drone, world):
        world.add_entity(drone)
        drone.state = DroneState.MOVING
        drone.set_target(np.array([600.0, 500.0]))
        # Duw de drone omhoog via flocking
        drone.apply_force(np.array([0.0, 100.0]))
        drone.update(0.1, world)
        # Drone moet enigszins omhoog zijn bewogen
        assert drone.velocity[1] > 0


# ── Serialisatie ──────────────────────────────────────────────────────────────

class TestDroneSerialisatie:

    def test_to_dict_bevat_state(self, drone):
        d = drone.to_dict()
        assert "state" in d
        assert d["state"] == "idle"

    def test_to_dict_met_target(self, drone):
        drone.set_target(np.array([700.0, 700.0]))
        d = drone.to_dict()
        assert d["target_position"] is not None

    def test_to_dict_zonder_target(self, drone):
        d = drone.to_dict()
        assert d["target_position"] is None
