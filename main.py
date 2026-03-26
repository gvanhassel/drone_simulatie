"""Entrypoint voor de drone-zwemsimulatie."""

from __future__ import annotations

import argparse
import os

from simulatie.simulation import Simulation
from simulatie.config import SimConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Drone Zwerm Simulatie")
    parser.add_argument("--drones", type=int, default=12, help="Aantal drones (default: 12)")
    parser.add_argument("--npcs", type=int, default=8, help="Aantal NPC's (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--headless", action="store_true", help="Draai zonder GUI")
    parser.add_argument("--steps", type=int, default=0,
                        help="Aantal stappen in headless modus (0=oneindig)")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.headless:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    sim = Simulation(
        sim_config=SimConfig(seed=args.seed),
        headless=args.headless,
    )
    sim.initialize(n_drones=args.drones, n_npcs=args.npcs)

    if args.headless and args.steps > 0:
        dt = 1.0 / 60
        for step in range(args.steps):
            sim.step(dt)
        metrics = sim.get_metrics()
        print(f"Klaar na {args.steps} stappen | {metrics}")
    else:
        sim.run()


if __name__ == "__main__":
    main()
