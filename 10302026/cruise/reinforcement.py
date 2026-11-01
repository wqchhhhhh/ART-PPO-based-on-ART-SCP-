"""PPO residual control for a fixed mission; no SCP or expert solve in training."""
from dataclasses import replace
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Normal
from cruise.config import CONTROL_SCALE, STATE_SCALE, Zone, bounds
from cruise.dynamics import Discretization
from cruise.geometry import great_circle, distance
from cruise.optimization import initial_guess, Solution
from cruise.validation import validate


def random_zones(settings, mission, count, seed):
    if count < 0: raise ValueError('zone count must be nonnegative')
    rng = np.random.default_rng(seed)
    zones = list(mission.zones)
    for _ in range(count):
        for attempt in range(1000):
            lon, lat = great_circle(*mission.initial[1:3], *mission.target[1:3],
                                   np.array([rng.uniform(.15, .85)]))
            lon, lat = float(lon[0]), float(lat[0])
            lat += rng.uniform(-150000, 150000)/settings.vehicle.radius
            lon += rng.uniform(-150000, 150000)/(settings.vehicle.radius*np.cos(lat))
            radius = rng.uniform(30, 90)
            if all(distance(lon, lat, p[1], p[2], settings.vehicle.radius) > radius*1000+settings.numerics.nfz_buffer
                   for p in (mission.initial, mission.target)):
                zones.append(Zone(float(np.degrees(lon)), float(np.degrees(lat)), float(radius)))
                break
        else: raise RuntimeError('Cannot sample endpoint-clear zones')
    return replace(mission, zones=tuple(zones))


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(15, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.actor, self.critic = nn.Linear(64, 3), nn.Linear(64, 1)
        nn.init.zeros_(self.actor.weight); nn.init.zeros_(self.actor.bias)
        self.log_std = nn.Parameter(torch.full((3,), -2.5))

    def forward(self, obs):
        h = self.body(obs)
        return Normal(self.actor(h), self.log_std.clamp(-5, 0).exp()), self.critic(h).squeeze(-1)


def optimize(settings, mission, updates=50, seed=42, warmstart=None, checkpoint=None):
    if updates < 1: raise ValueError('updates must be positive')
    torch.manual_seed(seed)
    started = time.perf_counter()
    disc = Discretization(settings, mission)
    n = settings.numerics.intervals
    reference, base = initial_guess(settings, mission)
    if warmstart:
        with np.load(warmstart, allow_pickle=False) as data:
            reference, base = data['states'].copy(), data['controls'].copy()
        if reference.shape != (n+1, 7) or base.shape != (n, 3):
            raise ValueError('Warm start horizon differs from mission')
        if not np.isfinite(reference).all() or not np.isfinite(base).all():
            raise ValueError('Warm start contains non-finite values')
    _, _, ulo, uhi = bounds(settings, mission)
    base = np.clip(base, ulo+1e-7, uhi-1e-7)
    logits = np.arctanh(np.clip(2*(base-ulo)/(uhi-ulo)-1, -.999, .999))
    model = Policy()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    def episode(sample):
        x = np.asarray(mission.initial, dtype=float)
        states, controls, records = [x.copy()], [], []
        for k in range(n):
            obs = torch.tensor(np.r_[mission.normalize(x), (x-reference[k])/STATE_SCALE, k/n], dtype=torch.float32)
            with torch.no_grad():
                dist, value = model(obs)
                action = dist.sample() if sample else dist.mean
                logp = dist.log_prob(action).sum()
            u = ulo+(uhi-ulo)*(np.tanh(logits[k]+action.numpy())+1)/2
            y = mission.normalize(x)
            with np.errstate(all='ignore'):
                nxt = mission.physical(np.asarray(disc.step(y, u/CONTROL_SCALE)).ravel())
                residual = np.asarray(disc.stage_functions[True](y, u/CONTROL_SCALE)[0]).ravel()
            invalid = not np.isfinite(nxt).all() or not np.isfinite(residual).all()
            if invalid:
                reward = -10000.
            else:
                violation = np.maximum(residual, 0)
                reward = -(x[6]-nxt[6])/1000 - 100*float(np.mean(violation**2))
                reward -= .02*float(np.sum(((nxt-reference[k+1])/STATE_SCALE)**2))/n
            controls.append(u); states.append(nxt.copy())
            terminal = invalid or k == n-1 or (np.isfinite(nxt).all() and (nxt[0] < 10000 or nxt[0] > 60000 or nxt[3] < 500 or nxt[6] < settings.vehicle.dry_mass))
            if terminal and not invalid:
                error = (nxt[:5]-mission.target)/STATE_SCALE[:5]
                reward -= 1000*float(error@error)
                if k != n-1: reward -= 10000
            records.append((obs, action, logp, value, reward, terminal))
            x = nxt
            if terminal: break
        return np.asarray(states), np.asarray(controls), records

    best, best_cost = None, float('inf')
    history = []
    def consider(x, u, records):
        nonlocal best, best_cost
        cost = -sum(r[4] for r in records)
        if len(u) == n and np.isfinite(x).all() and cost < best_cost:
            best_cost, best = cost, (x.copy(), u.copy())
        return cost
    x, u, rec = episode(False)
    consider(x, u, rec)
    for update in range(updates):
        samples = []
        for _ in range(4):
            x, u, records = episode(True)
            consider(x, u, records)
            advantage, last = [], 0.
            for i in reversed(range(len(records))):
                next_value = records[i+1][3].item() if i+1 < len(records) else 0.
                delta = records[i][4]+.995*next_value-records[i][3].item()
                last = delta+.995*.95*last
                advantage.append(last)
            advantage.reverse()
            samples.extend((r[0], r[1], r[2], a, a+r[3].item()) for r, a in zip(records, advantage))
        obs = torch.stack([s[0] for s in samples]); acts = torch.stack([s[1] for s in samples])
        old = torch.stack([s[2] for s in samples]); adv = torch.tensor([s[3] for s in samples])
        returns = torch.tensor([s[4] for s in samples]); adv = (adv-adv.mean())/(adv.std()+1e-8)
        for _ in range(4):
            dist, values = model(obs)
            ratio = (dist.log_prob(acts).sum(-1)-old).exp()
            loss = -torch.minimum(ratio*adv, ratio.clamp(.8, 1.2)*adv).mean()
            loss += .5*((values-returns)**2).mean() - .001*dist.entropy().sum(-1).mean()
            optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), .5); optimizer.step()
        x, u, rec = episode(False)
        cost = consider(x, u, rec)
        history.append(dict(update=update+1, cost=cost, best_cost=best_cost, steps=len(u)))
        print(f'PPO {update+1}/{updates}: cost={cost:.5g}, best={best_cost:.5g}', flush=True)
    if checkpoint:
        from pathlib import Path
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(policy=model.state_dict(), settings=settings.to_dict(), mission=mission.to_dict(),
                        reference_states=reference, reference_controls=base, seed=seed, history=history), checkpoint)
    if best is None:
        return Solution(x, u, False, 'rl_no_complete_rollout', time.perf_counter()-started, updates,
                        {'passed': False, 'failures': ['incomplete_rollout']}, history)
    x, u = best
    check = validate(settings, mission, x, u)
    return Solution(x, u, check['passed'], 'rl_validated' if check['passed'] else 'rl_infeasible',
                    time.perf_counter()-started, updates, check, history)
