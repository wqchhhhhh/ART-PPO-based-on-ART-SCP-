"""Forecasting and SCP comparisons with matched scenarios and solver settings."""

import json
from pathlib import Path
import time

import numpy as np

from cruise.config import Mission
from cruise.dataset import TrajectoryDataset, read_pair
from cruise.inference import rollout
from cruise.learning import load_model
from cruise.optimization import initial_guess, solve_expert, solve_scp


def benchmark(directory, checkpoint, output, limit=None, device="cpu", rtg_scale=1.):
    model, settings, saved = load_model(checkpoint, device)
    dataset = TrajectoryDataset(directory, "test", max_zones=model.config.max_zones, group_splits=saved["splits"])
    if dataset.fingerprint != saved["dataset_fingerprint"]:
        raise ValueError("Evaluation physical model differs from the checkpoint")
    test_ids = set(saved["splits"]["test"])
    rows = []
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    def persist():
        destination.write_text(json.dumps({"split": "test", "rtg_scale": rtg_scale, "scenarios": rows}, indent=2), encoding="utf-8")
    for path in dataset.files:
        data, meta = read_pair(path)
        mission = Mission.from_dict(meta["mission"])
        if mission.scenario_id not in test_ids: continue
        if limit is not None and len(rows) >= limit: break
        # The online relaxed solve provides a practical return condition. It is
        # a nonlinear relaxation for cruise; a lower bound is not certified.
        relaxed = solve_expert(settings, mission, include_nfz=False)
        if not relaxed.success:
            rows.append({"scenario_id": mission.scenario_id, "relaxed": relaxed.summary()})
            persist()
            continue
        target_return = -relaxed.validation["fuel_kg"]*rtg_scale
        row = {"scenario_id": mission.scenario_id, "rtg_kg": target_return,
               "reference_full": {"fuel_kg": meta["validation"][1]["fuel_kg"]},
               "relaxed": relaxed.summary(), "methods": {}}
        trajectories = {"relaxed_states": relaxed.states, "relaxed_controls": relaxed.controls,
                        "reference_states": data["states"][1], "reference_controls": data["controls"][1]}
        for name in ("geometry", "relaxed"):
            initialized = time.perf_counter()
            seed = initial_guess(settings, mission) if name == "geometry" else (relaxed.states, relaxed.controls)
            initialization_runtime = time.perf_counter()-initialized+(relaxed.runtime if name == "relaxed" else 0.)
            result = solve_scp(settings, mission, seed)
            summary = result.summary()
            summary["initialization_runtime"] = initialization_runtime
            summary["total_runtime"] = result.runtime+initialization_runtime
            row["methods"][name+"_scp"] = summary
            trajectories[name+"_scp_states"], trajectories[name+"_scp_controls"] = result.states, result.controls
        for mode in ("transformer", "dynamics"):
            try:
                predicted = rollout(model, settings, mission, target_return, mode=mode, device=device)
            except (RuntimeError, ValueError, FloatingPointError) as error:
                row["methods"]["art_"+mode] = {"success": False, "status": "generation_failed", "reason": str(error)}
                continue
            forecast = predicted.summary()
            forecast["state_rmse_si"] = np.sqrt(np.mean((predicted.states-data["states"][1])**2, axis=0)).tolist()
            forecast["control_rmse_si"] = np.sqrt(np.mean((predicted.controls-data["controls"][1])**2, axis=0)).tolist()
            forecast["forecast_terminal_error_si"] = (predicted.states[-1, :5]-mission.target).tolist()
            row["methods"]["art_"+mode] = forecast
            trajectories["art_"+mode+"_states"], trajectories["art_"+mode+"_controls"] = predicted.states, predicted.controls
            try:
                result = solve_scp(settings, mission, (predicted.states, predicted.controls))
            except (RuntimeError, ValueError, FloatingPointError) as error:
                row["methods"]["art_"+mode+"_scp"] = {"success": False, "status": "refinement_failed", "reason": str(error)}
                continue
            summary = result.summary()
            summary["total_runtime"] = relaxed.runtime+predicted.runtime+result.runtime
            row["methods"]["art_"+mode+"_scp"] = summary
            trajectories["art_"+mode+"_scp_states"], trajectories["art_"+mode+"_scp_controls"] = result.states, result.controls
        trajectory_path = destination.with_name(destination.stem+f"-{mission.scenario_id}.npz")
        np.savez_compressed(trajectory_path, **trajectories, mission=json.dumps(mission.to_dict()),
                            settings=json.dumps(settings.to_dict()))
        row["trajectory_file"] = trajectory_path.name
        rows.append(row)
        persist()
        print(json.dumps({"scenario_id": mission.scenario_id,
                          "success": {k: v["success"] for k, v in row["methods"].items()}}), flush=True)
    persist()
    return rows
