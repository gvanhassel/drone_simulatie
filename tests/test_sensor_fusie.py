"""Tests voor het sensorsysteem en SensorFusion."""

import numpy as np
import pytest

from simulatie.config import SimConfig, DroneConfig, SensorConfig
from simulatie.entities.drone import Drone, DroneState
from simulatie.entities.npc import NPC
from simulatie.sensoren.proximity import ProximitySensor
from simulatie.sensoren.fusie import SensorFusion
from simulatie.world import World


@pytest.fixture
def world():
    return World(SimConfig())


@pytest.fixture
def cfg():
    return DroneConfig()


# ── ProximitySensor tests ────────────────────────────────────────────────────

class TestProximitySensor:

    def test_detecteert_npc_in_bereik(self, world, cfg):
        drone = Drone(np.array([500.0, 500.0]), cfg)
        npc = NPC(np.array([600.0, 500.0]))  # 100px afstand, ruim binnen 200px bereik
        world.add_entity(drone)
        world.add_entity(npc)

        sensor = ProximitySensor(cfg)
        readings = sensor.scan(drone, world)
        entity_ids = [r.entity_id for r in readings]
        assert npc.id in entity_ids

    def test_detecteert_niet_buiten_bereik(self, world, cfg):
        drone = Drone(np.array([500.0, 500.0]), cfg)
        npc = NPC(np.array([800.0, 500.0]))  # 300px, buiten 200px bereik
        world.add_entity(drone)
        world.add_entity(npc)

        sensor = ProximitySensor(cfg)
        readings = sensor.scan(drone, world)
        entity_ids = [r.entity_id for r in readings]
        assert npc.id not in entity_ids

    def test_drone_detecteert_zichzelf_niet(self, world, cfg):
        drone = Drone(np.array([500.0, 500.0]), cfg)
        world.add_entity(drone)

        sensor = ProximitySensor(cfg)
        readings = sensor.scan(drone, world)
        drone_ids = [r.drone_id for r in readings]
        assert drone.id not in [r.entity_id for r in readings]

    def test_confidence_daalt_met_afstand(self, world, cfg):
        drone = Drone(np.array([500.0, 500.0]), cfg)
        npc_dichtbij = NPC(np.array([550.0, 500.0]))   # 50px
        npc_veraf = NPC(np.array([680.0, 500.0]))       # 180px
        world.add_entity(drone)
        world.add_entity(npc_dichtbij)
        world.add_entity(npc_veraf)

        sensor = ProximitySensor(cfg)
        readings = sensor.scan(drone, world)
        readings_by_id = {r.entity_id: r for r in readings}

        conf_dichtbij = readings_by_id[npc_dichtbij.id].confidence
        conf_veraf = readings_by_id[npc_veraf.id].confidence
        assert conf_dichtbij > conf_veraf

    def test_confidence_tussen_0_en_1(self, world, cfg):
        drone = Drone(np.array([500.0, 500.0]), cfg)
        npc = NPC(np.array([600.0, 500.0]))
        world.add_entity(drone)
        world.add_entity(npc)

        sensor = ProximitySensor(cfg)
        readings = sensor.scan(drone, world)
        for r in readings:
            assert 0.0 <= r.confidence <= 1.0


# ── SensorFusion tests ────────────────────────────────────────────────────────

class TestSensorFusion:

    @pytest.fixture
    def fusie(self):
        cfg = SensorConfig(grid_resolution=20, blur_sigma=0.5, decay_rate=0.85)
        return SensorFusion(2000, 2000, cfg)

    def test_output_shape_correct(self, fusie, world):
        drones = [Drone(np.array([500.0, 500.0]), DroneConfig())]
        for d in drones:
            world.add_entity(d)
        result = fusie.fuse_with_world(drones, world)
        assert result.shape == (fusie.grid_h, fusie.grid_w)

    def test_output_genormaliseerd_0_1(self, fusie, world):
        drone = Drone(np.array([500.0, 500.0]), DroneConfig())
        npc = NPC(np.array([600.0, 500.0]))
        world.add_entity(drone)
        world.add_entity(npc)
        result = fusie.fuse_with_world([drone], world)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0 + 1e-6

    def test_twee_drones_verhogen_waarden(self, world):
        cfg = SensorConfig(grid_resolution=20, blur_sigma=0.5, decay_rate=1.0)
        fusie_een = SensorFusion(2000, 2000, cfg)
        fusie_twee = SensorFusion(2000, 2000, cfg)

        npc = NPC(np.array([600.0, 500.0]))

        drone1 = Drone(np.array([500.0, 500.0]), DroneConfig())
        drone2 = Drone(np.array([520.0, 500.0]), DroneConfig())

        world2 = World(SimConfig())
        world2.add_entity(drone1)
        world2.add_entity(drone2)
        world2.add_entity(npc)

        world1 = World(SimConfig())
        world1.add_entity(Drone(np.array([500.0, 500.0]), DroneConfig()))
        world1.add_entity(NPC(np.array([600.0, 500.0])))

        result_een = fusie_een.fuse_with_world(world1.get_drones(), world1)
        result_twee = fusie_twee.fuse_with_world(world2.get_drones(), world2)

        # Twee drones die hetzelfde gebied zien → hogere ruwe accumulatie
        assert fusie_twee._accumulator.sum() >= fusie_een._accumulator.sum()

    def test_decay_verlaagt_waarden(self, fusie, world):
        drone = Drone(np.array([500.0, 500.0]), DroneConfig())
        npc = NPC(np.array([600.0, 500.0]))
        world.add_entity(drone)
        world.add_entity(npc)

        # Eerste scan vult accumulator
        fusie.fuse_with_world([drone], world)
        acc_na_eerste = fusie._accumulator.sum()

        # Tweede scan zonder NPC (verwijder npc)
        world.remove_entity(npc.id)
        fusie.fuse_with_world([drone], world)
        acc_na_tweede = fusie._accumulator.sum()

        assert acc_na_tweede < acc_na_eerste

    def test_reset_leegt_accumulator(self, fusie, world):
        drone = Drone(np.array([500.0, 500.0]), DroneConfig())
        npc = NPC(np.array([600.0, 500.0]))
        world.add_entity(drone)
        world.add_entity(npc)

        fusie.fuse_with_world([drone], world)
        fusie.reset()
        assert fusie._accumulator.sum() == 0.0

    def test_world_to_grid_grenzen(self, fusie):
        row, col = fusie.world_to_grid(np.array([0.0, 0.0]))
        assert row == 0 and col == 0

        row, col = fusie.world_to_grid(np.array([1999.0, 1999.0]))
        assert row == fusie.grid_h - 1 and col == fusie.grid_w - 1

    def test_lege_dronelijst_geeft_nullarray(self, fusie, world):
        result = fusie.fuse_with_world([], world)
        assert result.shape == (fusie.grid_h, fusie.grid_w)
        # Na één lege frame is accumulator × decay_rate ≈ 0
        assert float(result.max()) < 1e-6
