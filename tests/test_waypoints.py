"""Tests voor WaypointManager en OperatorInterface."""

import numpy as np
import pytest

from simulatie.config import DroneConfig, SimConfig
from simulatie.entities.drone import Drone, DroneState
from simulatie.world import World
from piloot.waypoints import Waypoint, WaypointManager
from piloot.interface import OperatorInterface, OperatorState


def maak_drone(pos):
    return Drone(np.array(pos, dtype=np.float32), DroneConfig())


# ── WaypointManager tests ────────────────────────────────────────────────────

class TestWaypointManager:

    @pytest.fixture
    def manager(self):
        return WaypointManager()

    def test_toevoegen_en_ophalen(self, manager):
        wp = manager.add_waypoint(np.array([500.0, 500.0]))
        assert manager.count == 1
        assert wp in manager.get_all()

    def test_verwijderen_op_id(self, manager):
        wp = manager.add_waypoint(np.array([500.0, 500.0]))
        manager.remove_waypoint(wp.id)
        assert manager.count == 0

    def test_verwijder_dichtstbijzijnde(self, manager):
        wp1 = manager.add_waypoint(np.array([100.0, 100.0]))
        wp2 = manager.add_waypoint(np.array([900.0, 900.0]))
        manager.remove_nearest(np.array([110.0, 110.0]))
        assert manager.count == 1
        assert wp2 in manager.get_all()

    def test_hongaarse_toewijzing_twee_drones_twee_waypoints(self, manager):
        """Hongaarse toewijzing moet minimale totale afstand geven."""
        # Drone 0 staat links, drone 1 rechts
        # Waypoint A staat rechts, waypoint B staat links
        # Optimaal: drone 0 → B, drone 1 → A (kruisend is NIET optimaal)
        drone0 = maak_drone([100, 500])
        drone1 = maak_drone([900, 500])
        manager.add_waypoint(np.array([900.0, 500.0]))  # Waypoint A (rechts)
        manager.add_waypoint(np.array([100.0, 500.0]))  # Waypoint B (links)

        manager.assign_drones([drone0, drone1])

        # Drone 0 moet naar links gaan (dichtstbij)
        assert drone0.target_position is not None
        assert drone0.target_position[0] < 500  # links

        # Drone 1 moet naar rechts gaan
        assert drone1.target_position is not None
        assert drone1.target_position[0] > 500  # rechts

    def test_aankomst_verwijdert_waypoint(self, manager):
        wp = manager.add_waypoint(np.array([500.0, 500.0]))
        drone = maak_drone([505.0, 500.0])  # Binnen arrival_radius=20px
        wp.assigned_drone_ids.append(drone.id)

        verwijderd = manager.check_arrivals([drone])
        assert len(verwijderd) == 1
        assert manager.count == 0

    def test_niet_gearriveerde_drone_behoudt_waypoint(self, manager):
        wp = manager.add_waypoint(np.array([500.0, 500.0]))
        drone = maak_drone([300.0, 300.0])  # Ver weg
        wp.assigned_drone_ids.append(drone.id)

        verwijderd = manager.check_arrivals([drone])
        assert len(verwijderd) == 0
        assert manager.count == 1

    def test_geen_waypoints_geen_fout(self, manager):
        drone = maak_drone([500.0, 500.0])
        manager.assign_drones([drone])  # Mag geen exception gooien
        manager.check_arrivals([drone])

    def test_clear_leegt_alles(self, manager):
        manager.add_waypoint(np.array([100.0, 100.0]))
        manager.add_waypoint(np.array([200.0, 200.0]))
        manager.clear()
        assert manager.count == 0


# ── OperatorInterface tests (headless) ───────────────────────────────────────

class TestOperatorInterface:

    @pytest.fixture
    def manager(self):
        return WaypointManager()

    @pytest.fixture
    def interface(self, manager):
        return OperatorInterface(waypoint_manager=manager)

    def test_waypoint_plaatsen_via_api(self, interface, manager):
        drone = maak_drone([500.0, 500.0])
        interface.place_waypoint(np.array([700.0, 700.0]), [drone])
        assert manager.count == 1
        assert drone.target_position is not None

    def test_waypoint_verwijderen_via_api(self, interface, manager):
        manager.add_waypoint(np.array([300.0, 300.0]))
        interface.remove_waypoint_at(np.array([310.0, 310.0]))
        assert manager.count == 0

    def test_initieel_niet_gepauzeerd(self, interface):
        assert interface.state.paused is False

    def test_initieel_heatmap_aan(self, interface):
        assert interface.state.show_heatmap is True
