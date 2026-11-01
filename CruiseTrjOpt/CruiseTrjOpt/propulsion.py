"""
Propulsion model module for Air-breathing Hypersonic Vehicle (AHV).
Models supersonic combustion ramjet (scramjet) installed thrust and fuel consumption.
Compatible with both CasADi symbolic structures and NumPy arrays.
"""

import casadi as ca
import numpy as np

import config as cfg


class Propulsion:
    """
    Scramjet propulsion system model.
    Thrust is scaled with dynamic pressure q and throttle ratio eta.
    """

    @staticmethod
    def max_thrust(q):
        """
        Compute maximum available scramjet thrust [N] at dynamic pressure q [Pa].
        T_max = CT_MAX * q * S_REF
        """
        return cfg.CT_MAX * q * cfg.S_REF

    @staticmethod
    def thrust(q, eta):
        """
        Compute actual scramjet thrust [N] given throttle ratio eta in [0, 1].
        """
        T_max = Propulsion.max_thrust(q)
        return eta * T_max

    @staticmethod
    def mass_flow_rate(T):
        """
        Compute fuel mass flow rate dm/dt [kg/s].
        dm/dt = - T / (Isp * g0)
        """
        return - T / (cfg.ISP * cfg.G0)
