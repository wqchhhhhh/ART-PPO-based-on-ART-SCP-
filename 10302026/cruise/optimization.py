"""Offline nonlinear experts and online SCP share dynamics and constraints."""

from dataclasses import dataclass, field
import time

import casadi as ca
import cvxpy as cp
import numpy as np

from cruise.config import CONTROL_SCALE, FUEL_SCALE, STATE_SCALE, bounds
from cruise.dynamics import Discretization, atmosphere, derivatives
from cruise.geometry import great_circle
from cruise.validation import validate


@dataclass
class Solution:
    states: np.ndarray
    controls: np.ndarray
    success: bool
    status: str
    runtime: float = 0.
    iterations: int = 0
    validation: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    converged: bool = False

    def summary(self):
        return {k: getattr(self, k) for k in ("success", "converged", "status", "runtime", "iterations", "validation", "history")}


def initial_guess(settings, mission, avoid_nfz=True):
    n = settings.numerics.intervals
    progress = np.linspace(0., 1., n+1)
    lon, lat = great_circle(mission.initial[1], mission.initial[2], mission.target[1], mission.target[2], progress)
    v = settings.vehicle
    if avoid_nfz:
        for zone in mission.zones:
            local_x = v.radius*np.cos(lat)*(lon-np.radians(zone.longitude_deg))
            local_y = v.radius*(lat-np.radians(zone.latitude_deg))
            k = int(np.argmin(local_x**2+local_y**2))
            if k in (0, n): continue
            dist = np.hypot(local_x[k], local_y[k])
            reserve = zone.radius_km*1400
            if dist >= reserve: continue
            tangent = np.array([np.cos(lat[k])*(lon[k+1]-lon[k-1]), lat[k+1]-lat[k-1]])
            normal = np.array([-tangent[1], tangent[0]])/np.linalg.norm(tangent)
            side = 1. if normal@np.array([local_x[k], local_y[k]]) >= 0 else -1.
            bump = side*(reserve-dist)*np.exp(-.5*((progress-progress[k])/.16)**2)*np.sin(np.pi*progress)
            lon += bump*normal[0]/(v.radius*np.cos(lat))
            lat += bump*normal[1]/v.radius
    x = np.tile(np.asarray(mission.initial), (n+1, 1))
    x[:, 0] = mission.initial[0]+progress*(mission.target[0]-mission.initial[0])
    x[:, 1], x[:, 2] = lon, lat
    x[:, 3] = mission.initial[3]+progress*(mission.target[3]-mission.initial[3])
    x[:, 4] = np.arcsin((mission.target[0]-mission.initial[0])/(mission.duration*x[:, 3]))
    x[:, 5] = np.unwrap(np.arctan2(np.cos(lat)*np.gradient(lon), np.gradient(lat)))
    x[:, 6] = mission.initial[6]-progress*min(6000., mission.initial[6]-v.dry_mass-100.)
    rho, _ = atmosphere(x[:, 0])
    q = .5*rho*x[:, 3]**2
    normal = x[:, 6]*(v.mu/(v.radius+x[:, 0])**2-x[:, 3]**2/(v.radius+x[:, 0]))
    alpha = (normal/(q*v.area)-v.cl0)/v.cl_alpha
    drag = q*v.area*(v.cd0+v.induced_drag*(v.cl0+v.cl_alpha*alpha)**2)
    u = np.column_stack([alpha, np.zeros(n+1), drag/(v.thrust_coefficient*q*v.area)])
    rates = np.gradient(x[:, 5], mission.duration/n)
    for k in range(n+1):
        base = derivatives(x[k], u[k], v)[5]
        gain = normal[k]/(x[k, 6]*x[k, 3]*np.cos(x[k, 4]))
        u[k, 1] = np.arcsin(np.clip((rates[k]-base)/gain, -.85, .85))
    x[0] = mission.initial
    x[-1, :5] = mission.target
    lo, hi, ulo, uhi = bounds(settings, mission)
    return np.clip(x, lo, hi), np.clip(u[:-1], ulo, uhi)


def solve_expert(settings, mission, include_nfz=True, warmstart=None, verbose=False):
    started = time.perf_counter()
    disc, n = Discretization(settings, mission), settings.numerics.intervals
    y, a = ca.MX.sym("states", 7, n+1), ca.MX.sym("controls", 3, n)
    objective = (mission.initial[6]-STATE_SCALE[6]*y[6, -1])/FUEL_SCALE
    equality = [y[:, 0]-mission.normalize(mission.initial),
                y[:5, -1]-mission.normalize([*mission.target, 0., 0.])[:5]]
    inequality = []
    for k in range(n):
        equality.append(y[:, k+1]-disc.step(y[:, k], a[:, k]))
        inequality.append(disc.stage_functions[include_nfz](y[:, k], a[:, k])[0])
    eq, ineq = ca.vertcat(*equality), ca.vertcat(*inequality)
    constraint = ca.vertcat(eq, ineq)
    nlp = {"x": ca.vertcat(ca.vec(y), ca.vec(a)), "f": objective, "g": constraint}
    lo, hi, ulo, uhi = bounds(settings, mission)
    lower = np.r_[np.tile(mission.normalize(lo), n+1), np.tile(ulo/CONTROL_SCALE, n)]
    upper = np.r_[np.tile(mission.normalize(hi), n+1), np.tile(uhi/CONTROL_SCALE, n)]
    seed_x, seed_u = warmstart if warmstart is not None else initial_guess(settings, mission, include_nfz)
    seed = np.r_[mission.normalize(seed_x).ravel(), (seed_u/CONTROL_SCALE).ravel()]
    solver = ca.nlpsol("expert", "ipopt", nlp, {"print_time": False,
        "ipopt": {"print_level": 5 if verbose else 0, "sb": "yes", "tol": 1e-8,
                  "constr_viol_tol": 1e-8, "max_iter": settings.numerics.expert_iterations,
                  "mu_strategy": "adaptive", "bound_relax_factor": 0.}})
    sol = solver(x0=np.clip(seed, lower, upper), lbx=lower, ubx=upper,
                 lbg=np.r_[np.zeros(eq.numel()), np.full(ineq.numel(), -np.inf)], ubg=0.)
    w = np.asarray(sol["x"]).ravel()
    x = mission.physical(w[:7*(n+1)].reshape(n+1, 7))
    u = w[7*(n+1):].reshape(n, 3)*CONTROL_SCALE
    stats = solver.stats()
    check = validate(settings, mission, x, u, include_nfz)
    return Solution(x, u, bool(stats["success"] and check["passed"]), stats["return_status"],
                    time.perf_counter()-started, stats["iter_count"], check, converged=bool(stats["success"]))


def solve_scp(settings, mission, warmstart=None, verbose=False):
    started = time.perf_counter()
    disc, num = Discretization(settings, mission), settings.numerics
    n = num.intervals
    seed_x, seed_u = initial_guess(settings, mission) if warmstart is None else warmstart
    ref_y, ref_a = mission.normalize(seed_x), np.asarray(seed_u)/CONTROL_SCALE
    lo, hi, ulo, uhi = bounds(settings, mission)
    target = mission.normalize([*mission.target, 0., 0.])[:5]
    radius = max(num.trust_initial, 1.05*float(np.max(np.abs(ref_y[-1, :5]-target))))
    penalty, history = num.penalty, []

    def evaluate(y, a):
        ends = np.stack([np.asarray(disc.step(y[k], a[k])).ravel() for k in range(n)])
        defects = y[1:]-ends
        g = np.concatenate([np.asarray(disc.stage_functions[True](y[k], a[k])[0]).ravel() for k in range(n)])
        endpoints = np.r_[y[0]-mission.normalize(mission.initial), y[-1, :5]-target]
        fuel = np.sum(y[:-1, 6]-ends[:, 6])*STATE_SCALE[6]/FUEL_SCALE
        violation = np.abs(defects).sum()+np.maximum(g, 0.).sum()+np.abs(endpoints).sum()
        return float(fuel), float(violation), float(np.max(np.abs(defects))), float(max(0., g.max()))

    def convex_step(restore=False):
        y, a = cp.Variable((n+1, 7)), cp.Variable((n, 3))
        virtual = cp.Variable((n, 7))
        constraints = [y[0] == mission.normalize(mission.initial), y[-1, :5] == target,
                       y >= mission.normalize(lo), y <= mission.normalize(hi),
                       a >= ulo/CONTROL_SCALE, a <= uhi/CONTROL_SCALE,
                       cp.abs(y-ref_y) <= radius, cp.abs(a-ref_a) <= radius]
        slack_sum, fuel = 0, 0
        for k in range(n):
            f, jac_x, jac_u = disc.linearize(ref_y[k], ref_a[k])
            affine = np.asarray(f).ravel()+np.asarray(jac_x)@(y[k]-ref_y[k])+np.asarray(jac_u)@(a[k]-ref_a[k])
            constraints.append(y[k+1] == affine+virtual[k])
            fuel += (y[k, 6]-affine[6])*STATE_SCALE[6]/FUEL_SCALE
            g, gx, gu = disc.stage_functions[True](ref_y[k], ref_a[k])
            slack = cp.Variable(g.numel(), nonneg=True)
            constraints.append(np.asarray(g).ravel()+np.asarray(gx)@(y[k]-ref_y[k])+np.asarray(gu)@(a[k]-ref_a[k]) <= slack)
            slack_sum += cp.sum(slack)
        # Exact-penalty merit has identical terms in the local and true models.
        local_cost = penalty*(cp.norm1(virtual)+slack_sum)
        local_cost += (cp.sum_squares(y-ref_y)+cp.sum_squares(a-ref_a)) if restore else fuel
        problem = cp.Problem(cp.Minimize(local_cost), constraints)
        try:
            problem.solve(solver="CLARABEL", max_iter=250, tol_gap_abs=1e-8,
                          tol_feas=1e-8, tol_gap_rel=1e-8, canon_backend=cp.SCIPY_CANON_BACKEND)
        except cp.error.SolverError:
            return None, "convex_solver_error"
        if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return None, problem.status
        return (y.value, a.value, float(problem.value), float(np.max(np.abs(virtual.value)))), problem.status

    status, converged = "iteration_limit", False
    for iteration in range(num.scp_iterations):
        base_fuel, base_violation, _, _ = evaluate(ref_y, ref_a)
        candidate, solver_status = convex_step()
        if candidate is None:
            radius *= .5
            history.append({"iteration": iteration+1, "accepted": False, "status": solver_status,
                            "radius": radius})
            if radius < num.trust_min:
                status = "trust_region_exhausted"
                break
            continue
        cand_y, cand_a, model_value, virtual_max = candidate
        new_fuel, new_violation, defect, path_violation = evaluate(cand_y, cand_a)
        predicted = base_fuel+penalty*base_violation-model_value
        actual = base_fuel+penalty*base_violation-new_fuel-penalty*new_violation
        rho = actual/predicted if predicted > 1e-10 else 0.
        feasible = defect < num.defect_tolerance and path_violation < 1e-7
        step = max(np.max(np.abs(cand_y-ref_y)), np.max(np.abs(cand_a-ref_a)))
        stationary = feasible and (abs(new_fuel-base_fuel) < num.cost_tolerance or step < num.step_tolerance)
        accepted = bool(np.isfinite(new_violation) and (rho >= .05 or stationary))
        row = {"iteration": iteration+1, "accepted": accepted, "fuel_kg": float(new_fuel*FUEL_SCALE),
               "defect": defect, "path_violation": path_violation, "rho": float(rho),
               "radius": radius, "virtual_max": virtual_max,
               "predicted_decrease": float(predicted), "actual_decrease": float(actual)}
        history.append(row)
        if verbose: print(row, flush=True)
        if accepted:
            ref_y, ref_a = cand_y, cand_a
            if stationary:
                status, converged = "converged", True
                break
            if rho > .75 and step > .8*radius: radius = min(2., radius*1.5)
        if not accepted or rho < .25: radius *= .5
        if radius < num.trust_min:
            status = "trust_region_exhausted"
            break
    # A few convex feasibility steps remove the final linearization error while
    # preserving a nearly feasible iterate. Cost stationarity is reported separately.
    _, violation, defect, path_violation = evaluate(ref_y, ref_a)
    if not converged and defect < 1e-3 and path_violation < 1e-3:
        radius = max(radius, .01)
        for _ in range(3):
            candidate, _ = convex_step(restore=True)
            if candidate is None: break
            cand_y, cand_a, _, virtual_max = candidate
            fuel, new_violation, defect, path_violation = evaluate(cand_y, cand_a)
            accepted = new_violation < violation
            history.append({"iteration": len(history)+1, "phase": "feasibility",
                            "accepted": bool(accepted), "fuel_kg": fuel*FUEL_SCALE,
                            "defect": defect, "path_violation": path_violation,
                            "virtual_max": virtual_max, "radius": radius})
            if not accepted: break
            ref_y, ref_a, violation = cand_y, cand_a, new_violation
            if defect < 1e-9 and path_violation < 1e-8: break
    x, u = mission.physical(ref_y), ref_a*CONTROL_SCALE
    check = validate(settings, mission, x, u)
    _, _, final_defect, final_path = evaluate(ref_y, ref_a)
    success = bool(check["passed"] and final_defect < num.defect_tolerance and final_path < 1e-7)
    if success and not converged: status = "feasible_"+status
    return Solution(x, u, success, status,
                    time.perf_counter()-started, len(history), check, history, converged)
