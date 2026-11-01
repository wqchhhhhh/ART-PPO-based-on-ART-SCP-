"""Paired NFZ-relaxed/full experts, ART labels and scenario-grouped splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cruise.config import CONTROL_SCALE, FUEL_SCALE, MODEL_VERSION, Mission, Settings, Zone
from cruise.geometry import bearing, clearances, distance, great_circle

SCHEMA_VERSION = 1


def labels(states, mission, vehicle):
    """Paper Eqs. 4--6 and 21: negative fuel-to-go and violating timestep count."""
    x = np.asarray(states)
    step_fuel = x[:-1, 6]-x[1:, 6]
    rtg = np.r_[-np.cumsum(step_fuel[::-1])[::-1], 0.]
    clearance = clearances(x[:-1], mission, vehicle.radius)
    violations = np.any(clearance < 0., axis=-1) if clearance.size else np.zeros(len(step_fuel), bool)
    ctg = np.r_[np.cumsum(violations[::-1])[::-1], 0.].astype(float)
    return rtg, ctg


def sample_mission(settings, base, seed):
    """Randomize cruise endpoints, fuel load, duration and obstacle geometry."""
    rng = np.random.default_rng(seed)
    initial, target = np.array(base.initial), np.array(base.target)
    shift = np.radians(rng.uniform(-1., 1., 2))
    initial[1:3] += shift
    target[1:3] += shift+np.radians(rng.uniform(-.5, .5, 2))
    # Endpoint pressure and heat feasibility are checked jointly.
    for state in (initial, target):
        from cruise.dynamics import atmosphere
        for _ in range(100):
            h = rng.uniform(29200., 31500.)
            rho, _ = atmosphere(h)
            vmin = max(2020., np.sqrt(2*settings.limits.pressure[0]*1.01/rho))
            vmax = min(2180., np.sqrt(2*settings.limits.pressure[1]*.99/rho),
                       (.99*settings.limits.heat_max/(settings.vehicle.heat_coefficient*np.sqrt(rho/settings.vehicle.nose_radius)))**(1/3))
            if vmin < vmax:
                state[0], state[3] = h, rng.uniform(vmin, vmax)
                break
        else:
            raise ValueError("No admissible cruise endpoint in the sampling envelope")
    initial[6] = rng.uniform(19500., 20500.)
    initial[5] = bearing(initial[1], initial[2], target[1], target[2])+np.radians(rng.uniform(-1., 1.))
    route = distance(initial[1], initial[2], target[1], target[2], settings.vehicle.radius)
    duration = float(route/(.5*(initial[3]+target[3]))*rng.uniform(1.01, 1.06))
    zones = []
    count = int(rng.integers(1, 4))
    fractions = np.linspace(.25, .75, count)+rng.uniform(-.04, .04, count)
    lons, lats = great_circle(initial[1], initial[2], target[1], target[2], fractions)
    for lon, lat in zip(lons, lats):
        lateral = rng.uniform(-80000., 80000.)/settings.vehicle.radius
        zones.append(Zone(float(np.degrees(lon)), float(np.degrees(lat+lateral)),
                          float(rng.uniform(100., 180.))))
    return Mission(tuple(initial), tuple(target), duration, tuple(zones), int(seed))


def solve_pair(settings, mission):
    from cruise.optimization import solve_expert
    relaxed = solve_expert(settings, mission, include_nfz=False)
    if not relaxed.success:
        return None, {"relaxed": relaxed.summary()}
    # A geometry-aware seed avoids the zero-gradient center of a circular NFZ.
    strict = solve_expert(settings, mission, include_nfz=True)
    if not strict.success:
        return None, {"relaxed": relaxed.summary(), "full": strict.summary()}
    return (relaxed, strict), {"relaxed": relaxed.summary(), "full": strict.summary()}


def write_pair(path, settings, mission, pair):
    """Only independently validated trajectories enter the current schema."""
    if len(pair) != 2 or not all(s.success and s.validation.get("passed") for s in pair):
        raise ValueError("Both experts must pass continuous validation before export")
    r, c = zip(*(labels(s.states, mission, settings.vehicle) for s in pair))
    meta = {"schema": SCHEMA_VERSION, "model": MODEL_VERSION, "settings": settings.to_dict(),
            "mission": mission.to_dict(), "kinds": ["nfz_relaxed", "full"],
            "control_interpolation": "zero_order_hold", "units": "SI; angles in radians",
            "validation": [s.validation for s in pair]}
    # Passing an open file makes creation exclusive and avoids implicit suffixes.
    with Path(path).open("xb") as f:
        np.savez_compressed(f, states=np.stack([s.states for s in pair]),
                            controls=np.stack([s.controls for s in pair]),
                            rtgs=np.stack(r), ctgs=np.stack(c), metadata=json.dumps(meta))


def read_pair(path):
    with np.load(path, allow_pickle=False) as f:
        meta = json.loads(str(f["metadata"]))
        if meta["schema"] != SCHEMA_VERSION or meta["model"] != MODEL_VERSION:
            raise ValueError("Dataset schema/model differs from the current cruise model")
        if not all(v["passed"] for v in meta["validation"]):
            raise ValueError("Dataset contains an unvalidated trajectory")
        arrays = {k: f[k] for k in ("states", "controls", "rtgs", "ctgs")}
        n = meta["settings"]["numerics"]["intervals"]
        expected = {"states": (2, n+1, 7), "controls": (2, n, 3), "rtgs": (2, n+1), "ctgs": (2, n+1)}
        for name, shape in expected.items():
            if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
                raise ValueError(f"Invalid current-schema tensor: {name}")
        return arrays, meta


def split_groups(group_ids, seed=42):
    groups = np.unique(group_ids)
    if len(groups) < 3:
        raise ValueError("At least three independent missions are needed for train/validation/test splits")
    groups = np.random.default_rng(seed).permutation(groups)
    nt = max(1, int(np.ceil(.1*len(groups))))
    nv = max(1, int(np.ceil(.1*(len(groups)-nt))))
    return {"test": set(groups[:nt].tolist()), "val": set(groups[nt:nt+nv].tolist()),
            "train": set(groups[nt+nv:].tolist())}


def context_vector(settings, mission, max_zones=3):
    """Explicit adaptation for variable goals, horizons and NFZ layouts."""
    if len(mission.zones) > max_zones:
        raise ValueError("Mission exceeds the model's configured maximum zone count")
    # Only constrained target components are provided; final mass is an outcome.
    target = mission.normalize([*mission.target, 0., 0.])[:5]
    values = [*target, mission.duration/2000.]
    for i in range(max_zones):
        if i < len(mission.zones):
            z = mission.zones[i]
            values.extend([(np.radians(z.longitude_deg)-mission.initial[1])/.1,
                           (np.radians(z.latitude_deg)-mission.initial[2])/.1,
                           z.radius_km/200., 1.])
        else:
            values.extend([0., 0., 0., 0.])
    return np.array(values, np.float32)


class TrajectoryDataset(Dataset):
    """Lazy per-mission shards; paired variants always share their split."""

    def __init__(self, directory, split="train", seed=42, max_zones=3, group_splits=None):
        self.files = sorted(Path(directory).glob("mission-*.npz"))
        if not self.files: raise ValueError("No current-schema mission shards found")
        self.entries, ids, fingerprint = [], [], None
        for path in self.files:
            arrays, meta = read_pair(path)
            signature = json.dumps(meta["settings"], sort_keys=True)
            if fingerprint is not None and signature != fingerprint:
                raise ValueError("All training shards must use the same model and discretization")
            fingerprint = signature
            ids.append(meta["mission"]["scenario_id"])
        if len(set(ids)) != len(ids): raise ValueError("Scenario IDs must be unique across shards")
        self.settings = Settings.from_dict(meta["settings"])
        self.group_splits = split_groups(ids, seed) if group_splits is None else {k: set(v) for k, v in group_splits.items()}
        if set().union(*self.group_splits.values()) != set(ids):
            raise ValueError("Saved split missions differ from this dataset")
        if any(self.group_splits[a] & self.group_splits[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise ValueError("Dataset splits overlap")
        for path, scenario_id in zip(self.files, ids):
            if split == "all" or scenario_id in self.group_splits[split]:
                self.entries.extend([(path, 0), (path, 1)])
        self.max_zones = max_zones
        self.fingerprint = hashlib.sha256(fingerprint.encode()).hexdigest()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        path, variant = self.entries[index]
        data, meta = read_pair(path)
        m = Mission.from_dict(meta["mission"])
        x = m.normalize(data["states"][variant])
        a = data["controls"][variant]/CONTROL_SCALE
        values = {"states": x[:-1], "next_states": x[1:], "actions": a,
                  "rtgs": data["rtgs"][variant, :-1, None]/FUEL_SCALE,
                  "ctgs": data["ctgs"][variant, :-1, None]/len(a),
                  "context": context_vector(self.settings, m, self.max_zones)}
        result = {k: torch.tensor(v, dtype=torch.float32) for k, v in values.items()}
        result["timesteps"] = torch.arange(len(a))
        return result
