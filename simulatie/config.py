"""Centrale configuratielaag voor de drone zwerm simulatie."""

from dataclasses import dataclass, field


@dataclass
class SimConfig:
    """Algemene simulatie-instellingen."""
    world_width: int = 2000      # pixels
    world_height: int = 2000     # pixels
    target_fps: int = 60         # frames per seconde
    seed: int = 42               # random seed voor reproduceerbaarheid
    spatial_cell_size: int = 100 # pixels per spatiale grid-cel


@dataclass
class DroneConfig:
    """Fysische parameters van een drone."""
    max_speed: float = 150.0         # px/s
    max_acceleration: float = 80.0   # px/s²
    max_turn_rate: float = 3.14      # rad/s
    sensor_range: float = 200.0      # px
    sensor_fov: float = 6.28318      # rad (360°)
    collision_radius: float = 10.0   # px
    arrival_radius: float = 20.0     # px — afstand waarop waypoint als bereikt geldt


@dataclass
class FlockingConfig:
    """Gewichten en radii voor Reynolds flocking algoritme."""
    separation_weight: float = 1.8
    alignment_weight: float = 1.0
    cohesion_weight: float = 1.2
    waypoint_weight: float = 2.5
    separation_radius: float = 50.0  # px — te dichtbij = separatiekracht
    neighbor_radius: float = 150.0   # px — buurradius voor uitlijning/coherentie


@dataclass
class SensorConfig:
    """Parameters voor sensorfusie en heatmap-generatie."""
    grid_resolution: int = 20    # wereldpixels per heatmap-gridcel
    blur_sigma: float = 1.5      # sterkte Gaussian blur
    decay_rate: float = 0.85     # heatmap-waarden vermenigvuldigd per frame


@dataclass
class RenderConfig:
    """Visuele weergave-instellingen."""
    screen_width: int = 1280     # pixels
    screen_height: int = 720     # pixels
    heatmap_panel_ratio: float = 0.70   # aandeel breedte voor heatmappanel
    show_sensor_radii: bool = False
    show_velocity_vectors: bool = True
    background_color: tuple = field(default_factory=lambda: (15, 15, 25))
