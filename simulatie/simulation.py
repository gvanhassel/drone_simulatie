"""Simulatie-orchestrator: koppelt alle subsystemen in de gameloop."""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict

import numpy as np

from simulatie.config import (
    SimConfig, DroneConfig, FlockingConfig, SensorConfig, RenderConfig
)
from simulatie.clock import SimClock
from simulatie.world import World
from simulatie.entities.drone import Drone
from simulatie.entities.npc import NPC
from simulatie.algoritmen.flocking import FlockingController
from simulatie.algoritmen.formaties import FormatieBeheerder
from simulatie.sensoren.fusie import SensorFusion
from piloot.waypoints import WaypointManager
from piloot.interface import OperatorInterface, OperatorState


class Simulation:
    """
    Centrale orchestrator voor de drone-zwemsimulatie.

    Vaste tick-volgorde per frame:
        1. Input verwerken
        2. Flocking krachten berekenen
        3. Krachten toepassen op drones
        4. Wereld stappen (alle entiteiten updaten)
        5. Sensor scan + fusie → heatmap data
        6. Waypoint aankomst controleren
        7. Frame renderen (skip in headless)
    """

    def __init__(
        self,
        sim_config: SimConfig | None = None,
        drone_config: DroneConfig | None = None,
        flocking_config: FlockingConfig | None = None,
        sensor_config: SensorConfig | None = None,
        render_config: RenderConfig | None = None,
        headless: bool = False,
    ) -> None:
        self.sim_cfg = sim_config or SimConfig()
        self.drone_cfg = drone_config or DroneConfig()
        self.flocking_cfg = flocking_config or FlockingConfig()
        self.sensor_cfg = sensor_config or SensorConfig()
        self.render_cfg = render_config or RenderConfig()
        self.headless = headless

        self.world = World(self.sim_cfg)
        self.clock = SimClock(self.sim_cfg.target_fps)
        self.flocking = FlockingController(self.flocking_cfg, self.drone_cfg)
        self.fusion = SensorFusion(
            self.sim_cfg.world_width,
            self.sim_cfg.world_height,
            self.sensor_cfg,
        )
        self.waypoint_manager = WaypointManager()
        self.operator_state = OperatorState(
            heatmap_panel_w=int(self.render_cfg.screen_width * self.render_cfg.heatmap_panel_ratio),
            heatmap_panel_h=self.render_cfg.screen_height,
        )
        self.operator_interface = OperatorInterface(
            state=self.operator_state,
            waypoint_manager=self.waypoint_manager,
        )
        self.formatie_beheerder = FormatieBeheerder()

        self._step_count: int = 0
        self._fusion_data: np.ndarray = np.zeros(
            (self.sim_cfg.world_height // self.sensor_cfg.grid_resolution,
             self.sim_cfg.world_width // self.sensor_cfg.grid_resolution),
            dtype=np.float32,
        )
        self._renderer = None
        self._rng = random.Random(self.sim_cfg.seed)
        self._running = False

    # ── Initialisatie ─────────────────────────────────────────────────────────

    def initialize(self, n_drones: int = 12, n_npcs: int = 8) -> None:
        """Maak drones en NPC's aan op willekeurige posities."""
        rng = self._rng
        w, h = self.sim_cfg.world_width, self.sim_cfg.world_height

        for _ in range(n_drones):
            pos = np.array([rng.uniform(100, w - 100), rng.uniform(100, h - 100)], dtype=np.float32)
            drone = Drone(pos, self.drone_cfg)
            self.world.add_entity(drone)

        for _ in range(n_npcs):
            pos = np.array([rng.uniform(50, w - 50), rng.uniform(50, h - 50)], dtype=np.float32)
            npc = NPC(pos, w, h, rng=random.Random(rng.randint(0, 999999)))
            self.world.add_entity(npc)

        if not self.headless:
            from interface.renderer import Renderer
            self._renderer = Renderer(
                self.render_cfg.screen_width,
                self.render_cfg.screen_height,
                self.render_cfg.heatmap_panel_ratio,
            )
            self._renderer.init()
            # Sla camera-referentie op in operator_state voor gebruik in interface
            self.operator_interface._camera = self._renderer.camera
            self.operator_state.heatmap_panel_w = self._renderer.heatmap_w

    # ── Stap ─────────────────────────────────────────────────────────────────

    def step(self, dt: float) -> None:
        """Voer één simulatiestap uit."""
        if self.operator_state.paused:
            return

        self._step_count += 1
        drones = self.world.get_drones()

        # Stap 2: flocking krachten
        forces = self.flocking.compute_forces(drones)

        # Formatie-override: als formatie actief, stel doel-posities in
        if self.formatie_beheerder.mode != 0 and drones:
            center = np.mean([d.position for d in drones], axis=0).astype(np.float32)
            doel_posities = self.formatie_beheerder.get_doel_posities(drones, center)
            if doel_posities:
                for drone in drones:
                    if drone.id in doel_posities:
                        drone.set_target(doel_posities[drone.id])

        # Stap 3: krachten toepassen
        for drone in drones:
            if drone.id in forces:
                drone.apply_force(forces[drone.id])

        # Stap 4: wereld stap
        self.world.step(dt)

        # Stap 5: sensor fusie
        self._fusion_data = self.fusion.fuse_with_world(drones, self.world)

        # Stap 6: waypoint aankomst
        self.waypoint_manager.check_arrivals(drones)

        # Formatiemodus synchroniseren via OperatorState
        if self.operator_state.formation_mode != self.formatie_beheerder.mode:
            self.formatie_beheerder.set_mode(self.operator_state.formation_mode)

    # ── Gameloop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Hoofd-eventloop. Stopt bij ESC of sluiten venster."""
        import pygame

        self._running = True
        target_dt = 1.0 / self.sim_cfg.target_fps

        while self._running:
            dt = self.clock.tick()

            # Stap 1: input
            if not self.headless:
                events = pygame.event.get()
                self.operator_interface.process_events(
                    events, self._renderer.camera, self.world.get_drones()
                )
                if self.operator_interface.quit_requested:
                    break

            self.step(dt)

            # Stap 7: renderen
            if not self.headless and self._renderer:
                self.operator_state.waypoints = self.waypoint_manager.get_all()
                self.operator_state.selected_drone_id = self.operator_state.selected_drone_id
                self._renderer.draw_frame(
                    self.world, self._fusion_data, self.operator_state, self.clock.fps
                )

            # Frame-rate beheersen in headless
            if self.headless:
                elapsed = self.clock._fps_samples[-1] if self.clock._fps_samples else 0
                sleep_time = target_dt - elapsed
                if sleep_time > 0.001:
                    time.sleep(sleep_time)

        if not self.headless and self._renderer:
            self._renderer.quit()

    # ── State opslaan/laden ───────────────────────────────────────────────────

    def save_state(self, path: str) -> None:
        """Sla de huidige simulatiestaat op als JSON."""
        state = {
            "drones": [d.to_dict() for d in self.world.get_drones()],
            "npcs": [n.to_dict() for n in self.world.get_npcs()],
            "frame": self.clock.frame_count,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    # ── Metrics ──────────────────────────────────────────────────────────────

    def get_metrics(self) -> dict:
        """Geeft huidige simulatiemetrics terug (voor tests en logging)."""
        drones = self.world.get_drones()
        return {
            "frame": self._step_count,
            "fps": self.clock.fps,
            "drone_count": len(drones),
            "npc_count": len(self.world.get_npcs()),
            "waypoint_count": self.waypoint_manager.count,
            "heatmap_max": float(self._fusion_data.max()),
        }
