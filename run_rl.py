"""PyCharm entry: PPO + additional random no-fly zones (no SCP)."""
import solver_environment
import argparse
from pathlib import Path
import os
from cruise.config import load_problem
from cruise.cli import save_solution
from cruise.reinforcement import random_zones, optimize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--updates', type=int, default=50)
    parser.add_argument('--extra-zones', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--intervals', type=int)
    parser.add_argument('--warmstart', help='ART/expert npz with states and controls; same horizon required')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent / '10302026'
    os.chdir(root)
    from dataclasses import replace
    settings, mission = load_problem('configs/cruise.yaml')
    if args.intervals:
        settings = replace(settings, numerics=replace(settings.numerics, intervals=args.intervals))
    mission = random_zones(settings, mission, args.extra_zones, args.seed)
    import json
    Path('results').mkdir(exist_ok=True)
    Path('results/rl_mission.json').write_text(json.dumps(mission.to_dict(), indent=2), encoding='utf-8')
    result = optimize(settings, mission, args.updates, args.seed, args.warmstart, 'checkpoints/ppo.pt')
    save_solution('results/rl_trajectory.npz', settings, mission, result)
    print(json.dumps(result.summary(), indent=2))
    if not result.success:
        print('Training finished; trajectory FAILED feasibility validation. Increase training or improve initialization.')
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
