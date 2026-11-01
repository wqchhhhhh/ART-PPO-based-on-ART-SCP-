"""Independent DOP853 replay of the stored zero-order-hold control sequence."""

import numpy as np
from scipy.integrate import solve_ivp

from cruise.config import bounds
from cruise.dynamics import derivatives, forces
from cruise.geometry import clearances, distance


def replay(settings, mission, controls):
    controls = np.asarray(controls)
    dt = mission.duration/len(controls)
    current = np.asarray(mission.initial, float)
    times, states, actions, nodes = [0.], [current], [controls[0]], [current]
    for k, u in enumerate(controls):
        grid = np.linspace(k*dt, (k+1)*dt, max(2, int(np.ceil(dt/settings.numerics.validation_step))+1))
        sol = solve_ivp(lambda t, x: derivatives(x, u, settings.vehicle),
                        (grid[0], grid[-1]), current, method="DOP853", t_eval=grid,
                        rtol=1e-10, atol=np.array([1e-6, 1e-12, 1e-12, 1e-7, 1e-12, 1e-12, 1e-7]),
                        max_step=settings.numerics.validation_step)
        if not sol.success:
            raise RuntimeError("Continuous replay failed: "+sol.message)
        # Both limits of a control switch participate in the path check.
        times.extend(grid.tolist())
        states.extend(sol.y.T)
        actions.extend(np.repeat(u[None, :], len(grid), axis=0))
        current = sol.y[:, -1]
        nodes.append(current)
    return np.asarray(times), np.asarray(states), np.asarray(actions), np.asarray(nodes)


def validate(settings, mission, states, controls, include_nfz=True):
    t, x, u, node_x = replay(settings, mission, controls)
    v, lim = settings.vehicle, settings.limits
    lo, hi, ulo, uhi = bounds(settings, mission)
    _, _, _, normal, q, heat, mach = forces(x.T, u.T, v)
    clearance = clearances(x, mission, v.radius)
    errors = np.max(np.abs(node_x-np.asarray(states)), axis=0)
    end = x[-1]
    miss = float(distance(end[1], end[2], mission.target[1], mission.target[2], v.radius))
    # Component tolerances are used only for independent numerical validation.
    # NLP endpoint equalities are solved to its own stricter tolerance.
    integration_tol = np.array([.2, 5e-8, 5e-8, .02, 2e-6, 2e-6, .02])
    initial_error = np.max(np.abs((np.asarray(states)[0]-mission.initial)/integration_tol))
    failures = []
    if initial_error > 1: failures.append("initial_state")
    if np.any(errors > integration_tol): failures.append("integration_consistency")
    if miss > 5.: failures.append("terminal_position")
    if abs(end[0]-mission.target[0]) > 2.: failures.append("terminal_altitude")
    if abs(end[3]-mission.target[3]) > .2: failures.append("terminal_speed")
    if abs(end[4]-mission.target[4]) > np.radians(.001): failures.append("terminal_gamma")
    state_tol = np.array([.2, 1e-8, 1e-8, .02, 2e-6, 2e-6, .02])
    if np.any(x < lo-state_tol) or np.any(x > hi+state_tol): failures.append("state_bounds")
    if np.any(u < ulo-1e-7) or np.any(u > uhi+1e-7): failures.append("control_bounds")
    if q.min() < lim.pressure[0]-.5 or q.max() > lim.pressure[1]+.5: failures.append("dynamic_pressure")
    if heat.max() > lim.heat_max+1.: failures.append("heat_flux")
    load = np.abs(normal)/(x[:, 6]*v.g0)
    if load.max() > lim.load_max+1e-5: failures.append("normal_load")
    min_clearance = float(clearance.min()) if clearance.size else None
    if include_nfz and min_clearance is not None and min_clearance < -1.: failures.append("no_fly_zone")
    return {"passed": not failures, "failures": failures, "nfz_enforced": include_nfz,
            "terminal_miss_m": miss, "terminal_altitude_error_m": float(end[0]-mission.target[0]),
            "terminal_speed_error_mps": float(end[3]-mission.target[3]),
            "terminal_gamma_error_deg": float(np.degrees(end[4]-mission.target[4])),
            "max_integration_error_si": errors.tolist(), "min_clearance_m": min_clearance,
            "fuel_kg": float(mission.initial[6]-end[6]),
            "pressure_range_pa": [float(q.min()), float(q.max())],
            "heat_max_w_m2": float(heat.max()), "load_max": float(load.max()),
            "mach_range": [float(mach.min()), float(mach.max())],
            "validation_step_s": settings.numerics.validation_step}
