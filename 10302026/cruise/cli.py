"""One entry point for expert generation, ART training and fair evaluation."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import torch

from cruise.config import load_problem
from cruise.dataset import read_pair, sample_mission, solve_pair, write_pair
from cruise.optimization import solve_expert, solve_scp


def save_solution(path, settings, mission, result):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, states=result.states, controls=result.controls,
                        settings=json.dumps(settings.to_dict()), mission=json.dumps(mission.to_dict()))
    path.with_suffix(".json").write_text(json.dumps(result.summary(), indent=2), encoding="utf-8")


def generate_worker(settings, base, seed, path):
    mission = sample_mission(settings, base, seed)
    pair, report = solve_pair(settings, mission)
    if pair is not None: write_pair(path, settings, mission, pair)
    return {"seed": seed, "accepted": pair is not None, "report": report}


def generate(args, settings, mission):
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = destination/"generation.jsonl"
    accepted = 0
    # An existing shard is validated before reuse; its seed is never overwritten.
    for path in destination.glob("mission-*.npz"):
        _, meta = read_pair(path)
        if meta["settings"] != json.loads(json.dumps(settings.to_dict())):
            raise ValueError("Existing generation settings differ from the requested settings")
        accepted += 1
    attempts, limit = 0, args.max_attempts or args.count*5
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        next_seed = args.seed
        while accepted < args.count:
            while len(pending) < min(args.workers, args.count-accepted) and attempts < limit:
                path = destination/f"mission-{next_seed:08d}.npz"
                seed = next_seed
                next_seed += 1
                if path.exists(): continue
                pending[pool.submit(generate_worker, settings, mission, seed, str(path))] = seed
                attempts += 1
            if not pending: break
            completed = next(as_completed(pending))
            seed = pending.pop(completed)
            report = completed.result()
            accepted += int(report["accepted"])
            with manifest.open("a", encoding="utf-8") as f: f.write(json.dumps(report)+"\n")
            print(f"seed={seed}, accepted={report['accepted']}, missions={accepted}/{args.count}", flush=True)
    if accepted < args.count:
        raise RuntimeError(f"Generated {accepted}/{args.count} mission pairs in {attempts} attempts; see {manifest}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    solve = commands.add_parser("solve", help="solve one cruise problem")
    solve.add_argument("--method", choices=["expert", "relaxed", "scp"], default="expert")
    solve.add_argument("--output", default="results/cruise.npz")
    solve.add_argument("--verbose", action="store_true")
    gen = commands.add_parser("generate", help="generate paired relaxed/full experts")
    gen.add_argument("--count", type=int, default=100)
    gen.add_argument("--workers", type=int, default=4)
    gen.add_argument("--seed", type=int, default=100)
    gen.add_argument("--max-attempts", type=int)
    gen.add_argument("--output", default="data/cruise")
    for command in (solve, gen):
        command.add_argument("--config", default="configs/cruise.yaml")
        command.add_argument("--intervals", type=int)
    training = commands.add_parser("train", help="teacher-forced ART training")
    training.add_argument("--data", default="data/cruise")
    training.add_argument("--output", default="checkpoints/art.pt")
    training.add_argument("--epochs", type=int, default=50)
    training.add_argument("--batch-size", type=int, default=4)
    training.add_argument("--accumulation", type=int, default=8)
    training.add_argument("--learning-rate", type=float, default=3e-5)
    training.add_argument("--layers", type=int, default=6)
    training.add_argument("--heads", type=int, default=6)
    training.add_argument("--embedding", type=int, default=384)
    training.add_argument("--context", type=int, default=100)
    training.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    training.add_argument("--seed", type=int, default=42)
    evaluation = commands.add_parser("evaluate", help="compare baseline/ART warm starts on held-out missions")
    evaluation.add_argument("--data", default="data/cruise")
    evaluation.add_argument("--checkpoint", default="checkpoints/art.pt")
    evaluation.add_argument("--output", default="results/benchmark.json")
    evaluation.add_argument("--limit", type=int)
    evaluation.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluation.add_argument("--rtg-scale", type=float, default=1.)
    plotting = commands.add_parser("plot", help="plot a saved cruise solution")
    plotting.add_argument("solution")
    plotting.add_argument("--output", default="results/cruise.png")
    args = parser.parse_args(argv)
    if args.command == "plot":
        from cruise.plotting import plot_solution
        plot_solution(args.solution, args.output)
    elif args.command == "train":
        from cruise.dataset import TrajectoryDataset
        from cruise.learning import train
        from cruise.model import ModelConfig
        dataset = TrajectoryDataset(args.data)
        config = ModelConfig(args.layers, args.heads, args.embedding, args.context,
                             dataset.settings.numerics.intervals)
        train(args.data, args.output, config, args.epochs, args.batch_size,
              args.learning_rate, args.accumulation, args.device, args.seed)
    elif args.command == "evaluate":
        from cruise.evaluation import benchmark
        benchmark(args.data, args.checkpoint, args.output, args.limit, args.device, args.rtg_scale)
    else:
        settings, mission = load_problem(args.config)
        if args.intervals is not None:
            settings = replace(settings, numerics=replace(settings.numerics, intervals=args.intervals))
        if args.command == "generate":
            generate(args, settings, mission)
        else:
            result = (solve_scp(settings, mission, verbose=args.verbose) if args.method == "scp"
                      else solve_expert(settings, mission, args.method == "expert", verbose=args.verbose))
            save_solution(args.output, settings, mission, result)
            print(json.dumps(result.summary(), indent=2))
            if not result.success: raise SystemExit(1)
