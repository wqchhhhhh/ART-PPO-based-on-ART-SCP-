"""US76 stratosphere in geometric altitude, with smooth layer joins.

The US76 layer formulas are evaluated in geopotential height. A 100 m tanh
transition joins the 20 km and 32 km layers for NLP derivatives.
The approximation error in density is below 0.02% over 20--40 km.
"""

import casadi as ca
import numpy as np


class Atmosphere:
    GEOPOTENTIAL_RADIUS = 6356766.0
    GAMMA_AIR = 1.4
    R_GAS = 287.05287
    G0 = 9.80665
    JOIN_WIDTH = 100.0
    T20 = 216.65
    P20 = 5474.8886696777745
    T32 = 228.65
    P32 = P20 * (T20 / T32) ** (G0 / (R_GAS * 0.001))

    @classmethod
    def properties(cls, h):
        """Return (temperature [K], pressure [Pa], density [kg/m³])."""
        symbolic = isinstance(h, (ca.SX, ca.MX, ca.DM))
        log, exp, tanh = (ca.log, ca.exp, ca.tanh) if symbolic else (np.log, np.exp, np.tanh)
        H = cls.GEOPOTENTIAL_RADIUS * h / (cls.GEOPOTENTIAL_RADIUS + h)
        t_low = cls.T20 + 0.001 * (H - 20000.0)
        t_high = cls.T32 + 0.0028 * (H - 32000.0)
        lp_low = np.log(cls.P20) + cls.G0 / (cls.R_GAS * 0.001) * log(cls.T20 / t_low)
        lp_high = np.log(cls.P32) + cls.G0 / (cls.R_GAS * 0.0028) * log(cls.T32 / t_high)
        lp_below = np.log(cls.P20) - cls.G0 / (cls.R_GAS * cls.T20) * (H - 20000.0)
        blend20 = 0.5 * (1.0 + tanh((H - 20000.0) / cls.JOIN_WIDTH))
        t_low = cls.T20 + blend20 * (t_low - cls.T20)
        lp_low = lp_below + blend20 * (lp_low - lp_below)
        blend = 0.5 * (1.0 + tanh((H - 32000.0) / cls.JOIN_WIDTH))
        temperature = t_low + blend * (t_high - t_low)
        pressure = exp(lp_low + blend * (lp_high - lp_low))
        return temperature, pressure, pressure / (cls.R_GAS * temperature)

    @classmethod
    def density(cls, h):
        return cls.properties(h)[2]

    @classmethod
    def temperature(cls, h):
        return cls.properties(h)[0]

    @classmethod
    def speed_of_sound(cls, h):
        temperature = cls.temperature(h)
        sqrt = ca.sqrt if isinstance(h, (ca.SX, ca.MX, ca.DM)) else np.sqrt
        return sqrt(cls.GAMMA_AIR * cls.R_GAS * temperature)

    @classmethod
    def mach_number(cls, h, V):
        return V / cls.speed_of_sound(h)

    @classmethod
    def dynamic_pressure(cls, h, V):
        return 0.5 * cls.density(h) * V ** 2
