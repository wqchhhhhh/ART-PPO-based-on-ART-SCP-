"""
Aerodynamic model module for Air-breathing Hypersonic Vehicle (AHV).
Computes lift, drag, aerodynamic coefficients, normal load factor, and stagnation heat flux.
Fully compatible with both CasADi symbolic variables and NumPy numeric arrays.
"""

import casadi as ca
import numpy as np

import config as cfg


class Aerodynamics:
    """
    Waverider model with linear alpha lift and a quadratic drag polar.
    """

    @staticmethod
    def lift_coefficient(alpha):
        """
        Compute lift coefficient CL = CL0 + CL_alpha * alpha
        alpha in radians.
        """
        return cfg.CL0 + cfg.CL_ALPHA * alpha

    @staticmethod
    def drag_coefficient(CL):
        """
        Compute drag coefficient CD = CD0 + K_IND * CL^2 (drag polar).
        """
        return cfg.CD0 + cfg.K_IND * (CL ** 2)

    @staticmethod
    def forces(q, alpha):
        """
        Compute aerodynamic lift L and drag D [N].
        q: dynamic pressure [Pa]
        alpha: angle of attack [rad]
        """
        CL = Aerodynamics.lift_coefficient(alpha)
        CD = Aerodynamics.drag_coefficient(CL)
        L = q * cfg.S_REF * CL
        D = q * cfg.S_REF * CD
        return L, D, CL, CD

    @staticmethod
    def load_factor(normal_force, mass):
        """
        Normal load factor |L + T*sin(alpha)| / (mass*g0), including thrust.
        """
        aero_force_sq = normal_force ** 2
        if isinstance(aero_force_sq, (ca.SX, ca.MX)):
            total_force = ca.sqrt(aero_force_sq + 1e-12)
        else:
            total_force = np.sqrt(aero_force_sq)
        return total_force / (mass * cfg.G0)

    @staticmethod
    def stagnation_heat_flux(rho, V):
        """
        Compute stagnation point convective heat flux [W/m^2]
        using Detra-Kemp-Riddell / Sutton-Graves formula:
        Q_dot = k_Q * sqrt(rho / R_N) * V^3
        """
        if isinstance(rho, (ca.SX, ca.MX)):
            sqrt_term = ca.sqrt(rho / cfg.NOSE_RADIUS + 1e-12)
        else:
            sqrt_term = np.sqrt(rho / cfg.NOSE_RADIUS)
        return cfg.K_HEAT * sqrt_term * (V ** 3)
