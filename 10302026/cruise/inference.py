"""ART Algorithms 1/2 with a shared action/cost/CTG convention."""

import time

import numpy as np
import torch

from cruise.config import CONTROL_SCALE, FUEL_SCALE, bounds
from cruise.dataset import context_vector
from cruise.dynamics import Discretization
from cruise.geometry import clearances
from cruise.optimization import Solution
from cruise.validation import validate


@torch.no_grad()
def rollout(model, settings, mission, target_rtg, target_ctg=0., mode="dynamics", device="cpu"):
    started = time.perf_counter()
    if mode not in ("dynamics", "transformer"):
        raise ValueError("Inference mode must be dynamics or transformer")
    n = settings.numerics.intervals
    if n > model.config.max_steps: raise ValueError("Horizon exceeds checkpoint timestep embeddings")
    model = model.to(device).eval()
    disc = Discretization(settings, mission)
    _, _, ulo, uhi = bounds(settings, mission)
    task = torch.tensor(context_vector(settings, mission, model.config.max_zones), device=device)[None]
    states, actions = [np.asarray(mission.initial)], []
    rtgs, ctgs = [float(target_rtg)], [float(target_ctg)]
    generation_started = time.perf_counter()
    for k in range(n):
        first = max(0, k+1-model.config.context_length)
        kwargs = {"states": torch.tensor(mission.normalize(states[first:]), dtype=torch.float32, device=device)[None],
                  "actions": torch.tensor(np.asarray([*actions, np.zeros(3)])[first:]/CONTROL_SCALE,
                                          dtype=torch.float32, device=device)[None],
                  "rtgs": torch.tensor(np.asarray(rtgs[first:])[:, None]/FUEL_SCALE, dtype=torch.float32, device=device)[None],
                  "ctgs": torch.tensor(np.asarray(ctgs[first:])[:, None]/n, dtype=torch.float32, device=device)[None],
                  "timesteps": torch.arange(first, k+1, device=device)[None], "context": task}
        pred_a, _ = model(**kwargs)
        u = np.clip(pred_a[0, -1].cpu().numpy()*CONTROL_SCALE, ulo, uhi)
        actions.append(u)
        physical_next = mission.physical(np.asarray(disc.step(mission.normalize(states[-1]), u/CONTROL_SCALE)).ravel())
        if mode == "transformer":
            kwargs["actions"][0, -1] = torch.tensor(u/CONTROL_SCALE, dtype=torch.float32, device=device)
            _, pred_x = model(**kwargs)
            next_x = mission.physical(pred_x[0, -1].cpu().numpy())
        else:
            next_x = physical_next
        # R_{k+1} = R_k - reward_k = R_k + fuel_k, in kilograms.
        rtgs.append(rtgs[-1]+states[-1][6]-physical_next[6])
        clearance = clearances(states[-1], mission, settings.vehicle.radius)
        ctgs.append(ctgs[-1]-float(np.any(clearance < 0.)))
        if not np.isfinite(next_x).all():
            raise RuntimeError(f"ART generated a non-finite state at interval {k}")
        states.append(next_x)
    generation_time = time.perf_counter()-generation_started
    x, u = np.array(states), np.array(actions)
    check = validate(settings, mission, x, u)
    check.update(initial_rtg_kg=float(target_rtg), final_rtg_kg=float(rtgs[-1]),
                 initial_ctg=float(target_ctg), final_ctg=float(ctgs[-1]),
                 generation_runtime=generation_time)
    return Solution(x, u, check["passed"], "art_"+mode, time.perf_counter()-started, n, check)
