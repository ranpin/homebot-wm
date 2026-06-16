"""Collect expert demonstrations from MuJoCo environment.

Usage:
    python scripts/collect_data.py --num_episodes 2000 --output data/trajectories.h5
"""

import argparse
from pathlib import Path

from wm_sim.data_collection import collect_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect expert trajectories")
    parser.add_argument("--num_episodes", type=int, default=2000)
    parser.add_argument("--image_size", type=int, default=84)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--output", type=str, default="data/trajectories.h5")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    collect_dataset(
        output_path=Path(args.output),
        num_episodes=args.num_episodes,
        image_size=args.image_size,
        max_steps=args.max_steps,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
