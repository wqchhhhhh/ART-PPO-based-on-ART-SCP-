"""
Initial guess generation for Air-breathing Hypersonic Vehicle (AHV) Trajectory Optimization.
Generates waypoint-based initial state and control estimates on LGL nodes.
The optimizer and independent propagation determine final feasibility.
"""

import numpy as np
from scipy.interpolate import CubicSpline

import config as cfg
from collocation import lgl_nodes_weights_diff
from atmosphere import Atmosphere


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Compute great circle distance [m] between two points on Earth.
    Inputs in radians.
    """
    dphi = lat2 - lat1
    dlam = lon2 - lon1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * cfg.R_E * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def generate_scenario_waypoints(p0, pf, nfzs):
    """
    Automatically generate intermediate waypoints that push the path outside
    any interfering no-fly zones along the flight corridor.
    p0: (lon_deg, lat_deg)
    pf: (lon_deg, lat_deg)
    nfzs: list of NFZ dicts
    """
    lon0, lat0 = p0
    lonf, latf = pf
    dlon = lonf - lon0
    dlat = latf - lat0
    norm = np.hypot(dlon, dlat)
    
    # Perpendicular unit vector (n_lon, n_lat)
    n_lon = - dlat / (norm + 1e-12)
    n_lat =   dlon / (norm + 1e-12)

    wps = [[lon0, lat0]]
    for nfz in nfzs:
        # Vector from start to NFZ center
        v_lon = nfz["lon_deg"] - lon0
        v_lat = nfz["lat_deg"] - lat0
        proj = (v_lon * dlon + v_lat * dlat) / (norm ** 2 + 1e-12)

        # If NFZ lies between start and target
        if 0.08 < proj < 0.92:
            c_line_lon = lon0 + proj * dlon
            c_line_lat = lat0 + proj * dlat
            
            # Cross product to determine which side of the line NFZ is on
            cross = dlon * (nfz["lat_deg"] - lat0) - dlat * (nfz["lon_deg"] - lon0)
            side = -1.0 if cross >= 0 else 1.0

            # Push waypoint to the opposite side of the center by (radius + margin)
            r_deg = (nfz["radius_km"] / 111.0) * 1.35
            wp_lon = c_line_lon + side * n_lon * r_deg
            wp_lat = c_line_lat + side * n_lat * r_deg
            wps.append([wp_lon, wp_lat])

    wps.append([lonf, latf])
    # Sort along flight direction
    wps = sorted(wps, key=lambda p: (p[0] - lon0) * dlon + (p[1] - lat0) * dlat)
    return np.array(wps)


def build_initial_guess(N_nodes, scenario_config=None):
    """
    Construct initial guess vectors for the LGL pseudospectral optimization problem.
    Supports default config or custom scenario_config dictionary.
    
    Returns:
        guess: dict containing tau, time, tf, states, controls, etc.
    """
    tau, w, D = lgl_nodes_weights_diff(N_nodes)

    if scenario_config is None:
        lon0_deg = cfg.LON_0_DEG
        lat0_deg = cfg.LAT_0_DEG
        lonf_deg = cfg.LON_F_DEG
        latf_deg = cfg.LAT_F_DEG
        nfzs = cfg.NO_FLY_ZONES
        alt_cruise = cfg.ALT_CRUISE
        vel_cruise = cfg.VEL_CRUISE
        m_init = cfg.M_INIT
    else:
        lon0_deg = scenario_config.get("lon_0_deg", cfg.LON_0_DEG)
        lat0_deg = scenario_config.get("lat_0_deg", cfg.LAT_0_DEG)
        lonf_deg = scenario_config.get("lon_f_deg", cfg.LON_F_DEG)
        latf_deg = scenario_config.get("lat_f_deg", cfg.LAT_F_DEG)
        nfzs = scenario_config.get("no_fly_zones", cfg.NO_FLY_ZONES)
        alt_cruise = scenario_config.get("alt_cruise", cfg.ALT_CRUISE)
        vel_cruise = scenario_config.get("vel_cruise", cfg.VEL_CRUISE)
        m_init = scenario_config.get("m_init", cfg.M_INIT)

    # 1. Generate geometry-aware waypoints
    wps_deg = generate_scenario_waypoints((lon0_deg, lat0_deg), (lonf_deg, latf_deg), nfzs)
    wps_lon_rad = np.radians(wps_deg[:, 0])
    wps_lat_rad = np.radians(wps_deg[:, 1])

    # Compute cumulative distance
    s_cum = [0.0]
    for i in range(len(wps_deg) - 1):
        seg_dist = great_circle_distance(
            wps_lon_rad[i], wps_lat_rad[i],
            wps_lon_rad[i + 1], wps_lat_rad[i + 1]
        )
        s_cum.append(s_cum[-1] + seg_dist)
    s_cum = np.array(s_cum)
    total_ground_dist = s_cum[-1]

    # Estimated flight time at nominal speed
    tf_guess = total_ground_dist / vel_cruise

    # Parameterize LGL nodes onto physical time and arc length
    t_nodes = (tf_guess / 2.0) * (tau + 1.0)
    s_nodes = (total_ground_dist / 2.0) * (tau + 1.0)

    # Cubic spline interpolation for longitude and latitude
    cs_lon = CubicSpline(s_cum, wps_lon_rad, bc_type="natural")
    cs_lat = CubicSpline(s_cum, wps_lat_rad, bc_type="natural")

    lon_nodes = cs_lon(s_nodes)
    lat_nodes = cs_lat(s_nodes)

    # Compute horizontal velocity components and heading angle psi
    dlon_ds = cs_lon(s_nodes, 1)
    dlat_ds = cs_lat(s_nodes, 1)

    v_east = cfg.R_E * np.cos(lat_nodes) * dlon_ds * vel_cruise
    v_north = cfg.R_E * dlat_ds * vel_cruise
    psi_nodes = np.arctan2(v_east, v_north)

    # Constant altitude, speed, and flight path angle
    h_nodes = np.full(N_nodes + 1, alt_cruise)
    r_nodes = cfg.R_E + h_nodes
    V_nodes = np.full(N_nodes + 1, vel_cruise)
    gamma_nodes = np.zeros(N_nodes + 1)

    # Fuel consumption and mass decay estimation
    q_nom = Atmosphere.dynamic_pressure(alt_cruise, vel_cruise)
    g_nom = cfg.MU_E / (r_nodes[0] ** 2)
    m_avg = 0.5 * (m_init + m_init - 6400.0)
    weight_eff = m_avg * (g_nom - (vel_cruise ** 2) / r_nodes[0])

    CL_trim = weight_eff / (q_nom * cfg.S_REF)
    alpha_trim = (CL_trim - cfg.CL0) / cfg.CL_ALPHA
    CD_trim = cfg.CD0 + cfg.K_IND * (CL_trim ** 2)
    D_trim = q_nom * cfg.S_REF * CD_trim

    T_max_nom = cfg.CT_MAX * q_nom * cfg.S_REF
    eta_trim = np.clip(D_trim / T_max_nom, 0.1, 0.95)

    mdot_trim = - D_trim / (cfg.ISP * cfg.G0)
    delta_mass = - mdot_trim * tf_guess
    m_nodes = m_init - delta_mass * (tau + 1.0) / 2.0

    # Angle of attack and throttle
    alpha_nodes = np.full(N_nodes + 1, alpha_trim)
    eta_nodes = np.full(N_nodes + 1, eta_trim)

    # Estimate bank angle from dpsi/dt = (2 / tf) * D @ psi
    dpsi_dt = (2.0 / tf_guess) * (D @ psi_nodes)
    sin_sigma = np.clip((dpsi_dt * m_nodes * V_nodes) / (weight_eff + 1e-5), -0.7, 0.7)
    sigma_nodes = np.arcsin(sin_sigma)

    # Pack states and controls
    states = np.column_stack([
        r_nodes,
        lon_nodes,
        lat_nodes,
        V_nodes,
        gamma_nodes,
        psi_nodes,
        m_nodes
    ])

    controls = np.column_stack([
        alpha_nodes,
        sigma_nodes,
        eta_nodes
    ])

    guess = {
        "tau": tau,
        "weights": w,
        "D": D,
        "time": t_nodes,
        "tf": tf_guess,
        "states": states,
        "controls": controls,
        "total_dist_km": total_ground_dist / 1000.0,
    }

    return guess
