"""
Three-Degree-of-Freedom (3-DOF) Point-Mass Dynamics on a Spherical Rotating Earth.
Formulates the state equations of motion for an Air-breathing Hypersonic Vehicle (AHV).
"""

import casadi as ca
import numpy as np

import config as cfg
from atmosphere import Atmosphere
from aerodynamics import Aerodynamics
from propulsion import Propulsion


def compute_dynamics(x, u):
    """
    Compute state derivatives dx/dt given state vector x and control vector u.
    
    States:
        x[0]: r      - Radial distance from Earth center [m]
        x[1]: theta  - Longitude [rad]
        x[2]: phi    - Latitude [rad]
        x[3]: V      - Relative velocity [m/s]
        x[4]: gamma  - Flight path angle [rad]
        x[5]: psi    - Heading angle (azimuth clockwise from North) [rad]
        x[6]: m      - Vehicle mass [kg]
        
    Controls:
        u[0]: alpha  - Angle of attack [rad]
        u[1]: sigma  - Bank angle [rad]
        u[2]: eta    - Throttle ratio [0, 1]
        
    Returns:
        xdot: array/SX vector of length 7
        aux: dict containing auxiliary variables (q, n, Q_dot, L, D, T, Ma, h)
    """
    r     = x[0]
    theta = x[1]
    phi   = x[2]
    V     = x[3]
    gamma = x[4]
    psi   = x[5]
    m     = x[6]

    alpha = u[0]
    sigma = u[1]
    eta   = u[2]

    # Altitude and atmospheric properties
    h = r - cfg.R_E
    rho = Atmosphere.density(h)
    q = 0.5 * rho * (V ** 2)
    a_sound = Atmosphere.speed_of_sound(h)
    Ma = V / a_sound

    # Aerodynamic forces
    L, D, CL, CD = Aerodynamics.forces(q, alpha)
    q_heat = Aerodynamics.stagnation_heat_flux(rho, V)

    # Propulsion forces
    T = Propulsion.thrust(q, eta)
    dm_dt = Propulsion.mass_flow_rate(T)
    normal_force = L + T * (ca.sin(alpha) if isinstance(alpha, (ca.SX, ca.MX, ca.DM)) else np.sin(alpha))
    n_load = Aerodynamics.load_factor(normal_force, m)

    # Gravitational acceleration
    g = cfg.MU_E / (r ** 2)

    # Trigonometric values
    if isinstance(r, (ca.SX, ca.MX, ca.DM)):
        sin_gam = ca.sin(gamma)
        cos_gam = ca.cos(gamma)
        tan_gam = ca.tan(gamma)
        sin_psi = ca.sin(psi)
        cos_psi = ca.cos(psi)
        sin_phi = ca.sin(phi)
        cos_phi = ca.cos(phi)
        tan_phi = ca.tan(phi)
        sin_alp = ca.sin(alpha)
        cos_alp = ca.cos(alpha)
        sin_sig = ca.sin(sigma)
        cos_sig = ca.cos(sigma)
    else:
        sin_gam = np.sin(gamma)
        cos_gam = np.cos(gamma)
        tan_gam = np.tan(gamma)
        sin_psi = np.sin(psi)
        cos_psi = np.cos(psi)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)
        tan_phi = np.tan(phi)
        sin_alp = np.sin(alpha)
        cos_alp = np.cos(alpha)
        sin_sig = np.sin(sigma)
        cos_sig = np.cos(sigma)

    # Earth rotation effects (Coriolis and Centrifugal accelerations)
    we = cfg.OMEGA_E
    we2_r = (we ** 2) * r

    # 1. dr/dt
    dr = V * sin_gam

    # 2. dtheta/dt (Longitude)
    dtheta = (V * cos_gam * sin_psi) / (r * cos_phi)

    # 3. dphi/dt (Latitude)
    dphi = (V * cos_gam * cos_psi) / r

    # 4. dV/dt (Velocity)
    dV = ((T * cos_alp - D) / m
          - g * sin_gam
          + we2_r * cos_phi * (sin_gam * cos_phi - cos_gam * sin_phi * cos_psi))

    # 5. dgamma/dt (Flight path angle)
    # Total effective lift including thrust vertical projection: L_eff = L + T * sin(alpha)
    L_eff = L + T * sin_alp
    dgamma = (1.0 / V) * (
        (L_eff * cos_sig) / m
        + ((V ** 2) / r - g) * cos_gam
        + 2.0 * we * V * cos_phi * sin_psi
        + we2_r * cos_phi * (cos_gam * cos_phi + sin_gam * sin_phi * cos_psi)
    )

    # 6. dpsi/dt (Heading angle, clockwise from North)
    dpsi = ((L_eff * sin_sig) / (m * V * cos_gam)
            + (V / r) * cos_gam * sin_psi * tan_phi
            - 2.0 * we * (tan_gam * cos_phi * cos_psi - sin_phi)
            + (we2_r / (V * cos_gam)) * sin_phi * cos_phi * sin_psi)

    # 7. dm/dt (Mass)
    dm = dm_dt

    if isinstance(r, (ca.SX, ca.MX, ca.DM)):
        xdot = ca.vertcat(dr, dtheta, dphi, dV, dgamma, dpsi, dm)
    else:
        xdot = np.array([dr, dtheta, dphi, dV, dgamma, dpsi, dm])

    aux = {
        "h": h,
        "rho": rho,
        "q": q,
        "Ma": Ma,
        "L": L,
        "D": D,
        "T": T,
        "CL": CL,
        "CD": CD,
        "n_load": n_load,
        "q_heat": q_heat,
    }

    return xdot, aux


def get_casadi_dynamics_function():
    """
    Construct CasADi Function for dynamics and path constraint quantities.
    Inputs:
        x (7x1)
        u (3x1)
    Outputs:
        xdot (7x1)
        path_outputs (4x1): [q, n_load, q_heat, Ma]
    """
    x_sym = ca.SX.sym("x", 7)
    u_sym = ca.SX.sym("u", 3)
    xdot_sym, aux_sym = compute_dynamics(x_sym, u_sym)
    
    path_sym = ca.vertcat(
        aux_sym["q"],
        aux_sym["n_load"],
        aux_sym["q_heat"],
        aux_sym["Ma"]
    )
    
    f_dyn = ca.Function("f_dynamics", [x_sym, u_sym], [xdot_sym, path_sym],
                        ["x", "u"], ["xdot", "path_vars"])
    return f_dyn


def compute_dynamics_reference(x, u):
    """Independent numeric force balance in rotating Earth-fixed Cartesian axes."""
    r,lon,lat,V,gamma,psi,mass = np.asarray(x,dtype=float)
    alpha,bank,throttle = np.asarray(u,dtype=float)
    er = np.array([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)])
    east = np.array([-np.sin(lon),np.cos(lon),0.])
    north = np.cross(er,east)
    horizontal = np.sin(psi)*east+np.cos(psi)*north
    velocity_unit = np.cos(gamma)*horizontal+np.sin(gamma)*er
    gamma_unit = -np.sin(gamma)*horizontal+np.cos(gamma)*er
    heading_unit = np.cos(psi)*east-np.sin(psi)*north
    q = Atmosphere.dynamic_pressure(r-cfg.R_E,V)
    lift,drag,_,_ = Aerodynamics.forces(q,alpha)
    thrust = Propulsion.thrust(q,throttle)
    force = ((thrust*np.cos(alpha)-drag)*velocity_unit
             +(lift+thrust*np.sin(alpha))*(np.cos(bank)*gamma_unit+np.sin(bank)*heading_unit))
    omega = np.array([0.,0.,cfg.OMEGA_E])
    acceleration = (force/mass-cfg.MU_E/r**2*er
                    -2*np.cross(omega,V*velocity_unit)-np.cross(omega,np.cross(omega,r*er)))
    return np.array([V*np.sin(gamma),V*np.cos(gamma)*np.sin(psi)/(r*np.cos(lat)),
                     V*np.cos(gamma)*np.cos(psi)/r,np.dot(acceleration,velocity_unit),
                     np.dot(acceleration,gamma_unit)/V+V*np.cos(gamma)/r,
                     np.dot(acceleration,heading_unit)/(V*np.cos(gamma))
                     +V*np.cos(gamma)*np.sin(psi)*np.tan(lat)/r,
                     -thrust/(cfg.ISP*cfg.G0)])
