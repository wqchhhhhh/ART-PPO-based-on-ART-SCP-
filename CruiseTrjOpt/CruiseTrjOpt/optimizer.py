"""
Trajectory Optimization Solver using CasADi and IPOPT.
LGL collocation supplies a seed, refined using piecewise-linear controls and
multiple shooting, then accepted by independent continuous-trajectory checks.
Supports customizable mission scenarios (start, target, NFZs, altitude, velocity).
"""

import time
import casadi as ca
import numpy as np

import config as cfg
from dynamics import get_casadi_dynamics_function
from constraints import Constraints
from collocation import lgl_nodes_weights_diff
from initial_guess import build_initial_guess


class TrajectoryOptimizer:
    """
    NLP formulation and solution for fuel-optimal cruise trajectory optimization.
    """

    def __init__(self, N_nodes=cfg.N_NODES, scenario_config=None):
        self.N_nodes = N_nodes
        self.N_points = N_nodes + 1
        self.tau, self.weights, self.D = lgl_nodes_weights_diff(N_nodes)
        self.f_dyn = get_casadi_dynamics_function()
        self.scenario = scenario_config or {}

        # Mission boundaries
        self.lon_0 = np.radians(self.scenario.get("lon_0_deg", cfg.LON_0_DEG))
        self.lat_0 = np.radians(self.scenario.get("lat_0_deg", cfg.LAT_0_DEG))
        self.lon_f = np.radians(self.scenario.get("lon_f_deg", cfg.LON_F_DEG))
        self.lat_f = np.radians(self.scenario.get("lat_f_deg", cfg.LAT_F_DEG))
        
        self.alt_0 = self.scenario.get("alt_0", cfg.ALT_0)
        self.alt_f = self.scenario.get("alt_f", cfg.ALT_F)
        self.vel_0 = self.scenario.get("vel_0", cfg.VEL_0)
        self.vel_f = self.scenario.get("vel_f", cfg.VEL_F)
        self.gamma_0 = self.scenario.get("gamma_0", cfg.GAMMA_0)
        self.gamma_f = self.scenario.get("gamma_f", cfg.GAMMA_F)
        self.m_init = self.scenario.get("m_init", cfg.M_INIT)

        raw_nfzs = self.scenario.get("no_fly_zones", cfg.NO_FLY_ZONES)
        self.nfzs = Constraints.prepare_nfzs(raw_nfzs)
        
        # Scaling vectors
        self.x_scales = np.array([
            cfg.SCALE_R,
            1.0,
            1.0,
            cfg.SCALE_V,
            cfg.SCALE_GAMMA,
            1.0,
            cfg.SCALE_MASS,
        ])
        
        self.u_scales = np.array([
            cfg.SCALE_ALPHA,
            cfg.SCALE_SIGMA,
            cfg.SCALE_ETA,
        ])
        
        self.tf_scale = cfg.SCALE_TF

    def build_nlp(self):
        """
        Construct NLP optimization problem symbolically in CasADi.
        """
        X_norm = ca.SX.sym("X_norm", self.N_points, 7)
        U_norm = ca.SX.sym("U_norm", self.N_points, 3)
        tf_norm = ca.SX.sym("tf_norm")

        tf = tf_norm * self.tf_scale
        
        w = []
        w_lb = []
        w_ub = []

        # Box bounds for states
        min_lon = min(self.lon_0, self.lon_f) - 0.15
        max_lon = max(self.lon_0, self.lon_f) + 0.15
        min_lat = min(self.lat_0, self.lat_f) - 0.15
        max_lat = max(self.lat_0, self.lat_f) + 0.15

        x_norm_lb = [
            cfg.ALT_MIN / cfg.SCALE_R,
            min_lon,
            min_lat,
            cfg.VEL_MIN / cfg.SCALE_V,
            cfg.GAMMA_MIN / cfg.SCALE_GAMMA,
            cfg.PSI_MIN,
            cfg.MASS_MIN / cfg.SCALE_MASS,
        ]
        x_norm_ub = [
            cfg.ALT_MAX / cfg.SCALE_R,
            max_lon,
            max_lat,
            cfg.VEL_MAX / cfg.SCALE_V,
            cfg.GAMMA_MAX / cfg.SCALE_GAMMA,
            cfg.PSI_MAX,
            self.m_init / cfg.SCALE_MASS,
        ]

        # Box bounds for controls
        u_norm_lb = [
            cfg.ALPHA_MIN / cfg.SCALE_ALPHA,
            cfg.SIGMA_MIN / cfg.SCALE_SIGMA,
            cfg.ETA_MIN / cfg.SCALE_ETA,
        ]
        u_norm_ub = [
            cfg.ALPHA_MAX / cfg.SCALE_ALPHA,
            cfg.SIGMA_MAX / cfg.SCALE_SIGMA,
            cfg.ETA_MAX / cfg.SCALE_ETA,
        ]

        # Flatten X_norm into w
        for i in range(self.N_points):
            for j in range(7):
                w.append(X_norm[i, j])
                w_lb.append(x_norm_lb[j])
                w_ub.append(x_norm_ub[j])

        # Flatten U_norm into w
        for i in range(self.N_points):
            for j in range(3):
                w.append(U_norm[i, j])
                w_lb.append(u_norm_lb[j])
                w_ub.append(u_norm_ub[j])

        # Add tf_norm into w
        w.append(tf_norm)
        w_lb.append(cfg.TF_MIN / self.tf_scale)
        w_ub.append(cfg.TF_MAX / self.tf_scale)

        w = ca.vertcat(*w)
        w_lb = np.array(w_lb)
        w_ub = np.array(w_ub)

        # Objective Function: Maximize final mass (minimize fuel consumed)
        final_mass_norm = X_norm[-1, 6]
        J_fuel = - final_mass_norm

        J_reg = 0.0
        w_reg = 1e-4
        for i in range(self.N_points - 1):
            d_alpha = U_norm[i + 1, 0] - U_norm[i, 0]
            d_sigma = U_norm[i + 1, 1] - U_norm[i, 1]
            d_eta   = U_norm[i + 1, 2] - U_norm[i, 2]
            J_reg += (d_alpha ** 2 + d_sigma ** 2 + d_eta ** 2)

        obj = J_fuel + w_reg * J_reg

        # Constraints g(w)
        g = []
        g_lb = []
        g_ub = []

        # A. Boundary Conditions
        # Altitude h(0) = alt_0
        g.append(X_norm[0, 0])
        g_lb.append(self.alt_0 / cfg.SCALE_R)
        g_ub.append(self.alt_0 / cfg.SCALE_R)

        # Longitude theta(0) = lon_0
        g.append(X_norm[0, 1])
        g_lb.append(self.lon_0)
        g_ub.append(self.lon_0)

        # Latitude phi(0) = lat_0
        g.append(X_norm[0, 2])
        g_lb.append(self.lat_0)
        g_ub.append(self.lat_0)

        # Velocity V(0) = vel_0
        g.append(X_norm[0, 3])
        g_lb.append(self.vel_0 / cfg.SCALE_V)
        g_ub.append(self.vel_0 / cfg.SCALE_V)

        # Flight path angle gamma(0) = gamma_0
        g.append(X_norm[0, 4])
        g_lb.append(self.gamma_0 / cfg.SCALE_GAMMA)
        g_ub.append(self.gamma_0 / cfg.SCALE_GAMMA)

        # Initial mass m(0) = m_init
        g.append(X_norm[0, 6])
        g_lb.append(self.m_init / cfg.SCALE_MASS)
        g_ub.append(self.m_init / cfg.SCALE_MASS)

        # Final states
        # Altitude h(tf) = alt_f
        g.append(X_norm[-1, 0])
        g_lb.append(self.alt_f / cfg.SCALE_R)
        g_ub.append(self.alt_f / cfg.SCALE_R)

        # Longitude theta(tf) = lon_f
        g.append(X_norm[-1, 1])
        g_lb.append(self.lon_f)
        g_ub.append(self.lon_f)

        # Latitude phi(tf) = lat_f
        g.append(X_norm[-1, 2])
        g_lb.append(self.lat_f)
        g_ub.append(self.lat_f)

        # Velocity V(tf) = vel_f
        g.append(X_norm[-1, 3])
        g_lb.append(self.vel_f / cfg.SCALE_V)
        g_ub.append(self.vel_f / cfg.SCALE_V)

        # Flight path angle gamma(tf) = gamma_f
        g.append(X_norm[-1, 4])
        g_lb.append(self.gamma_f / cfg.SCALE_GAMMA)
        g_ub.append(self.gamma_f / cfg.SCALE_GAMMA)

        # B. Collocation Defect Constraints: D @ X_norm - (tf / 2.0) * F_dot = 0
        D_ca = ca.DM(self.D)
        
        X_dot_norm = []
        path_vars_list = []

        for i in range(self.N_points):
            x_phys = ca.vertcat(
                cfg.R_E + X_norm[i, 0] * cfg.SCALE_R,
                X_norm[i, 1],
                X_norm[i, 2],
                X_norm[i, 3] * cfg.SCALE_V,
                X_norm[i, 4] * cfg.SCALE_GAMMA,
                X_norm[i, 5],
                X_norm[i, 6] * cfg.SCALE_MASS,
            )
            u_phys = ca.vertcat(
                U_norm[i, 0] * cfg.SCALE_ALPHA,
                U_norm[i, 1] * cfg.SCALE_SIGMA,
                U_norm[i, 2] * cfg.SCALE_ETA,
            )
            
            xdot_i, path_i = self.f_dyn(x_phys, u_phys)
            xdot_norm_i = ca.vertcat(
                xdot_i[0] / cfg.SCALE_R,
                xdot_i[1],
                xdot_i[2],
                xdot_i[3] / cfg.SCALE_V,
                xdot_i[4] / cfg.SCALE_GAMMA,
                xdot_i[5],
                xdot_i[6] / cfg.SCALE_MASS,
            )
            X_dot_norm.append(xdot_norm_i)
            path_vars_list.append(path_i)

        F_dot = ca.hcat(X_dot_norm).T
        defects = ca.mtimes(D_ca, X_norm) - (tf / 2.0) * F_dot

        for i in range(self.N_points):
            for j in range(7):
                g.append(defects[i, j])
                g_lb.append(0.0)
                g_ub.append(0.0)

        # C. Path Constraints & No-Fly Zone Constraints
        for i in range(self.N_points):
            theta_i = X_norm[i, 1]
            phi_i = X_norm[i, 2]
            
            # No-Fly Zone Exclusion
            nfz_residuals = Constraints.scaled_no_fly_zone_residuals(
                theta_i, phi_i, self.nfzs, cfg.NFZ_BUFFER_M)
            for k in range(len(self.nfzs)):
                g.append(nfz_residuals[k])
                g_lb.append(-np.inf)
                g_ub.append(0.0)

            # Dynamic pressure q [kPa]
            q_kpa = path_vars_list[i][0] / 1000.0
            g.append(q_kpa)
            g_lb.append(cfg.Q_MIN / 1000.0)
            g_ub.append(cfg.Q_MAX / 1000.0)

            # Aerodynamic normal load factor n
            n_load = path_vars_list[i][1]
            g.append(n_load)
            g_lb.append(0.0)
            g_ub.append(cfg.N_MAX)

            # Stagnation heat flux [kW/m^2]
            q_heat_kpa = path_vars_list[i][2] / 1000.0
            g.append(q_heat_kpa)
            g_lb.append(0.0)
            g_ub.append(cfg.Q_HEAT_MAX / 1000.0)

        g = ca.vertcat(*g)
        g_lb = np.array(g_lb)
        g_ub = np.array(g_ub)

        nlp = {"x": w, "f": obj, "g": g}
        return nlp, w_lb, w_ub, g_lb, g_ub

    def pack_initial_guess(self, guess_dict):
        states = guess_dict["states"]
        controls = guess_dict["controls"]
        tf = guess_dict["tf"]

        w0 = []
        for i in range(self.N_points):
            h_i = states[i, 0] - cfg.R_E
            r_norm = h_i / cfg.SCALE_R
            theta_i = states[i, 1]
            phi_i = states[i, 2]
            V_norm = states[i, 3] / cfg.SCALE_V
            gam_norm = states[i, 4] / cfg.SCALE_GAMMA
            psi_i = states[i, 5]
            m_norm = states[i, 6] / cfg.SCALE_MASS
            w0.extend([r_norm, theta_i, phi_i, V_norm, gam_norm, psi_i, m_norm])

        for i in range(self.N_points):
            alp_norm = controls[i, 0] / cfg.SCALE_ALPHA
            sig_norm = controls[i, 1] / cfg.SCALE_SIGMA
            eta_norm = controls[i, 2] / cfg.SCALE_ETA
            w0.extend([alp_norm, sig_norm, eta_norm])

        w0.append(tf / self.tf_scale)
        return np.array(w0)

    def unpack_solution(self, w_sol):
        w_arr = np.array(w_sol).flatten()
        
        states_flat = w_arr[:self.N_points * 7].reshape(self.N_points, 7)
        states = np.zeros_like(states_flat)
        states[:, 0] = cfg.R_E + states_flat[:, 0] * cfg.SCALE_R
        states[:, 1] = states_flat[:, 1]
        states[:, 2] = states_flat[:, 2]
        states[:, 3] = states_flat[:, 3] * cfg.SCALE_V
        states[:, 4] = states_flat[:, 4] * cfg.SCALE_GAMMA
        states[:, 5] = states_flat[:, 5]
        states[:, 6] = states_flat[:, 6] * cfg.SCALE_MASS

        u_start = self.N_points * 7
        u_end = u_start + self.N_points * 3
        controls_flat = w_arr[u_start:u_end].reshape(self.N_points, 3)
        controls = np.zeros_like(controls_flat)
        controls[:, 0] = controls_flat[:, 0] * cfg.SCALE_ALPHA
        controls[:, 1] = controls_flat[:, 1] * cfg.SCALE_SIGMA
        controls[:, 2] = controls_flat[:, 2] * cfg.SCALE_ETA

        tf = w_arr[-1] * self.tf_scale
        time_nodes = (tf / 2.0) * (self.tau + 1.0)

        altitudes = states[:, 0] - cfg.R_E
        machs = []
        dyn_pressures = []
        load_factors = []
        heat_fluxes = []

        for i in range(self.N_points):
            x_i = states[i, :]
            u_i = controls[i, :]
            _, aux_i = self.f_dyn(x_i, u_i)
            dyn_pressures.append(float(aux_i[0]))
            load_factors.append(float(aux_i[1]))
            heat_fluxes.append(float(aux_i[2]))
            machs.append(float(aux_i[3]))

        sol_dict = {
            "tau": self.tau,
            "weights": self.weights,
            "D": self.D,
            "time": time_nodes,
            "tf": tf,
            "states": states,
            "controls": controls,
            "altitude": altitudes,
            "longitude_deg": np.degrees(states[:, 1]),
            "latitude_deg": np.degrees(states[:, 2]),
            "velocity": states[:, 3],
            "gamma_deg": np.degrees(states[:, 4]),
            "psi_deg": np.degrees(states[:, 5]),
            "mass": states[:, 6],
            "alpha_deg": np.degrees(controls[:, 0]),
            "sigma_deg": np.degrees(controls[:, 1]),
            "throttle": controls[:, 2],
            "dynamic_pressure_kpa": np.array(dyn_pressures) / 1000.0,
            "load_factor": np.array(load_factors),
            "heat_flux_kw": np.array(heat_fluxes) / 1000.0,
            "mach": np.array(machs),
            "fuel_consumed_kg": states[0, 6] - states[-1, 6],
            "nfzs": self.nfzs,
            "scenario": self.scenario,
        }
        return sol_dict

    def _solve_lgl(self, max_iter=500, tol=1e-6, print_level=5):
        nlp, w_lb, w_ub, g_lb, g_ub = self.build_nlp()
        guess = build_initial_guess(self.N_nodes, scenario_config=self.scenario)
        w0 = self.pack_initial_guess(guess)

        opts = {
            "ipopt": {
                "max_iter": max_iter,
                "tol": tol,
                "constr_viol_tol": 1e-5,
                "bound_relax_factor": 0.0,
                "print_level": print_level,
                "linear_solver": "mumps",
                "nlp_scaling_method": "gradient-based",
                "mu_strategy": "adaptive",
                "hessian_approximation": "exact",
            },
            "print_time": (print_level > 0),
        }

        solver = ca.nlpsol("trj_solver", "ipopt", nlp, opts)
        t_start = time.time()
        res = solver(x0=w0, lbx=w_lb, ubx=w_ub, lbg=g_lb, ubg=g_ub)
        t_elapsed = time.time() - t_start

        stats = solver.stats()
        success = stats["success"]
        return_status = stats["return_status"]

        sol_dict = self.unpack_solution(res["x"])
        sol_dict["success"] = success
        sol_dict["return_status"] = return_status
        sol_dict["solve_time"] = t_elapsed
        return sol_dict

    def solve(self, max_iter=500, tol=1e-8, print_level=5):
        """LGL initialization, multiple-shooting refinement and continuous acceptance."""
        from shooting import refine_trajectory
        from continuous import validate_solution

        started = time.perf_counter()
        seed = self._solve_lgl(max_iter=max_iter, tol=tol, print_level=print_level)
        if not seed["success"]:
            seed["solver_success"] = False
            seed["validation"] = {"passed": False, "failures": ["LGL initialization failed"]}
            seed["solve_time"] = time.perf_counter() - started
            return seed
        initial_status = seed["return_status"]
        intervals = self.scenario.get("shooting_intervals", cfg.SHOOTING_INTERVALS)
        history = []
        for refinement in range(2):
            sol = refine_trajectory(self, seed, intervals=intervals,
                                    max_step=cfg.RK4_MAX_STEP / (2 ** refinement),
                                    max_iter=max_iter, tol=tol, print_level=print_level)
            if not sol["solver_success"]:
                sol["validation"] = {"passed": False, "failures": ["Multiple shooting failed"]}
                break
            validation = validate_solution(sol)
            sol["validation"] = validation
            history.append({"intervals": intervals, "rk4_max_step": cfg.RK4_MAX_STEP / (2 ** refinement),
                            "passed": validation["passed"], "failures": validation["failures"]})
            if validation["passed"]:
                break
            seed = sol
            if not self.scenario.get("fixed_control_grid",False):
                intervals *= 2
        sol["success"] = bool(sol["solver_success"] and sol["validation"]["passed"])
        sol["lgl_status"] = initial_status
        sol["refinement_history"] = history
        sol["solve_time"] = time.perf_counter() - started
        return sol
