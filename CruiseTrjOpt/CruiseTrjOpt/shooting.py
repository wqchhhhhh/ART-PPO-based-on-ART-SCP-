"""Sparse multiple shooting with bounded piecewise-linear controls.

The LGL solution supplies the initial guess. Each RK4 substep is constrained,
and the final result is accepted through independent DOP853 propagation.
"""

import time
import casadi as ca
import numpy as np

import config as cfg
from collocation import interpolate_trajectory
from constraints import Constraints


def refine_trajectory(optimizer, seed, intervals, max_step, max_iter, tol, print_level):
    M = int(intervals)
    substeps = int(np.ceil(cfg.TF_MAX / (M * max_step)))
    scales = ca.DM(optimizer.x_scales)
    offsets = ca.DM([cfg.R_E, 0, 0, 0, 0, 0, 0])
    u_scales = ca.DM(optimizer.u_scales)
    sx = ca.SX.sym("state", 7)
    su = ca.SX.sym("control", 3)
    derivative, path = optimizer.f_dyn(offsets + sx * scales, su * u_scales)
    f = ca.Function("scaled_dynamics", [sx, su], [derivative / scales, path])

    left = ca.SX.sym("left", 3)
    right = ca.SX.sym("right", 3)
    horizon = ca.SX.sym("horizon")
    dt = horizon * cfg.SCALE_TF / (M * substeps)
    current = sx
    stages, paths = [], []
    for j in range(substeps):
        u0 = left + (right - left) * (j / substeps)
        um = left + (right - left) * ((j + 0.5) / substeps)
        u1 = left + (right - left) * ((j + 1.0) / substeps)
        k1 = f(current, u0)[0]
        k2 = f(current + 0.5 * dt * k1, um)[0]
        k3 = f(current + 0.5 * dt * k2, um)[0]
        k4 = f(current + dt * k3, u1)[0]
        current = current + dt / 6.0 * (k1 + 2*k2 + 2*k3 + k4)
        stages.append(current)
        paths.append(f(current, u1)[1])
    segment = ca.Function("shooting_segment", [sx, left, right, horizon],
                          [current, ca.hcat(stages), ca.hcat(paths)])

    X = ca.MX.sym("shooting_states", 7, M + 1)
    U = ca.MX.sym("shooting_controls", 3, M + 1)
    tf = ca.MX.sym("shooting_tf")
    variables = ca.vertcat(ca.vec(X), ca.vec(U), tf)
    lo = np.array([cfg.ALT_MIN/cfg.SCALE_R, min(optimizer.lon_0,optimizer.lon_f)-.15,
                   min(optimizer.lat_0,optimizer.lat_f)-.15, cfg.VEL_MIN/cfg.SCALE_V,
                   cfg.GAMMA_MIN/cfg.SCALE_GAMMA, cfg.PSI_MIN, cfg.MASS_MIN/cfg.SCALE_MASS])
    hi = np.array([cfg.ALT_MAX/cfg.SCALE_R, max(optimizer.lon_0,optimizer.lon_f)+.15,
                   max(optimizer.lat_0,optimizer.lat_f)+.15, cfg.VEL_MAX/cfg.SCALE_V,
                   cfg.GAMMA_MAX/cfg.SCALE_GAMMA, cfg.PSI_MAX, optimizer.m_init/cfg.SCALE_MASS])
    ulo = np.array([cfg.ALPHA_MIN,cfg.SIGMA_MIN,cfg.ETA_MIN])/optimizer.u_scales
    uhi = np.array([cfg.ALPHA_MAX,cfg.SIGMA_MAX,cfg.ETA_MAX])/optimizer.u_scales
    vl = np.r_[np.tile(lo,M+1),np.tile(ulo,M+1),cfg.TF_MIN/cfg.SCALE_TF]
    vu = np.r_[np.tile(hi,M+1),np.tile(uhi,M+1),cfg.TF_MAX/cfg.SCALE_TF]
    boundary = {0:optimizer.alt_0/cfg.SCALE_R,1:optimizer.lon_0,2:optimizer.lat_0,
                3:optimizer.vel_0/cfg.SCALE_V,4:optimizer.gamma_0/cfg.SCALE_GAMMA,
                6:optimizer.m_init/cfg.SCALE_MASS}
    for k,v in boundary.items():
        vl[k] = vu[k] = v
    terminal = {0:optimizer.alt_f/cfg.SCALE_R,1:optimizer.lon_f,2:optimizer.lat_f,
                3:optimizer.vel_f/cfg.SCALE_V,4:optimizer.gamma_f/cfg.SCALE_GAMMA}
    for k,v in terminal.items():
        vl[M*7+k] = vu[M*7+k] = v

    constraints, gl, gu = [], [], []
    def add(value, lower, upper):
        n = int(value.numel())
        constraints.append(ca.vec(value))
        gl.extend(np.broadcast_to(lower,(n,)).tolist())
        gu.extend(np.broadcast_to(upper,(n,)).tolist())

    # Constrain all substep states and nonlinear path variables. Reserves cover
    # integration error and are independently checked at continuous extrema.
    for i in range(M):
        end, sample_x, sample_path = segment(X[:,i],U[:,i],U[:,i+1],tf)
        add(X[:,i+1]-end,0.,0.)
        progress = (i + np.arange(1,substeps+1)/substeps)/M
        reserve = np.sin(np.pi*progress)
        sl = np.tile(lo,(substeps,1))
        sh = np.tile(hi,(substeps,1))
        sl[:,0] += cfg.ALT_BUFFER_M/cfg.SCALE_R*reserve
        sh[:,0] -= cfg.ALT_BUFFER_M/cfg.SCALE_R*reserve
        sl[:,3] += cfg.VEL_BUFFER_MPS/cfg.SCALE_V*reserve
        sh[:,3] -= cfg.VEL_BUFFER_MPS/cfg.SCALE_V*reserve
        sl[:,4] += 1e-5/cfg.SCALE_GAMMA*reserve
        sh[:,4] -= 1e-5/cfg.SCALE_GAMMA*reserve
        add(sample_x,sl.ravel(),sh.ravel())
        b = cfg.PATH_BUFFER_FRACTION
        add(sample_path[0,:]/1000.,cfg.Q_MIN/1000.*(1+b),cfg.Q_MAX/1000.*(1-b))
        add(sample_path[1,:],0.,cfg.N_MAX*(1-b))
        add(sample_path[2,:]/1000.,0.,cfg.Q_HEAT_MAX/1000.*(1-b))
        if optimizer.nfzs:
            nfz = Constraints.scaled_no_fly_zone_residuals(
                sample_x[1,:].T,sample_x[2,:].T,optimizer.nfzs,cfg.NFZ_BUFFER_M)
            add(nfz,-np.inf,0.)
    initial_path = f(X[:,0],U[:,0])[1]
    add(initial_path[0]/1000.,cfg.Q_MIN/1000.,cfg.Q_MAX/1000.)
    add(initial_path[1],0.,cfg.N_MAX)
    add(initial_path[2]/1000.,0.,cfg.Q_HEAT_MAX/1000.)
    if optimizer.nfzs:
        add(Constraints.scaled_no_fly_zone_residuals(X[1,0],X[2,0],optimizer.nfzs),-np.inf,0.)

    rate_penalty = ca.sumsqr(U[:,1:]-U[:,:-1])/(tf/M)
    objective = -X[6,-1] + cfg.CONTROL_REG_WEIGHT*rate_penalty
    nlp = {"x":variables,"f":objective,"g":ca.vertcat(*constraints)}
    opts = {"ipopt":{"max_iter":max_iter,"tol":min(tol,1e-8),"constr_viol_tol":1e-8,
                     "acceptable_tol":1e-8,"bound_relax_factor":0.,"print_level":print_level,
                     "mu_strategy":"adaptive","linear_solver":"mumps"},
            "print_time":print_level>0}
    solver = ca.nlpsol("continuous_solver","ipopt",nlp,opts)
    t0 = np.linspace(0.,seed["tf"],M+1)
    x0 = interpolate_trajectory(seed["time"],seed["states"],t0)
    x0[:,0] -= cfg.R_E
    x0 /= optimizer.x_scales
    u0 = interpolate_trajectory(seed["time"],seed["controls"],t0)/optimizer.u_scales
    w0 = np.r_[x0.ravel(),u0.ravel(),seed["tf"]/cfg.SCALE_TF]
    started = time.perf_counter()
    result = solver(x0=w0,lbx=vl,ubx=vu,lbg=gl,ubg=gu)
    stats = solver.stats()
    w = np.array(result["x"]).ravel()
    states = w[:7*(M+1)].reshape(M+1,7)*optimizer.x_scales
    states[:,0] += cfg.R_E
    controls = w[7*(M+1):-1].reshape(M+1,3)*optimizer.u_scales
    duration = float(w[-1]*cfg.SCALE_TF)
    from continuous import describe_trajectory
    sol = describe_trajectory(np.linspace(0.,duration,M+1),states,controls,
                              optimizer.nfzs,optimizer.scenario)
    sol.update(success=bool(stats["success"]),solver_success=bool(stats["success"]),
               return_status=stats["return_status"],method="LGL seed + RK4 multiple shooting",
               control_interpolation="piecewise_linear",shooting_intervals=M,
               rk4_substeps=substeps,model_version=cfg.MODEL_VERSION,
               refinement_solve_time=time.perf_counter()-started)
    g = np.array(result["g"]).ravel()
    sol["nlp_constraint_violation"] = float(np.max(np.maximum(np.array(gl)-g,g-np.array(gu))))
    return sol
