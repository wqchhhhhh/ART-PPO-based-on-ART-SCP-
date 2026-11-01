"""Shared NumPy/CasADi cruise physics and zero-order-hold RK4 transition.

The US76 cruise atmosphere, drag polar and pressure-scaled thrust come from
CruiseTrjOpt. State zero is altitude throughout this package.
"""

import casadi as ca
import numpy as np

from cruise.config import CONTROL_SCALE, STATE_SCALE, bounds


def symbolic(value):
    return isinstance(value, (ca.SX, ca.MX, ca.DM))


def atmosphere(h):
    """Smooth US76 layers spanning the configured 25--35 km cruise corridor."""
    log, exp, tanh = (ca.log, ca.exp, ca.tanh) if symbolic(h) else (np.log, np.exp, np.tanh)
    geop = 6356766.*h/(6356766.+h)
    gas, g0, t20, p20 = 287.05287, 9.80665, 216.65, 5474.8886696777745
    t32 = 228.65
    p32 = p20*(t20/t32)**(g0/(gas*.001))
    tlow, thigh = t20+.001*(geop-20000.), t32+.0028*(geop-32000.)
    plow = np.log(p20)+g0/(gas*.001)*log(t20/tlow)
    phigh = np.log(p32)+g0/(gas*.0028)*log(t32/thigh)
    pbelow = np.log(p20)-g0/(gas*t20)*(geop-20000.)
    b20 = .5*(1+tanh((geop-20000.)/100.))
    tlow = t20+b20*(tlow-t20)
    plow = pbelow+b20*(plow-pbelow)
    b32 = .5*(1+tanh((geop-32000.)/100.))
    temp = tlow+b32*(thigh-tlow)
    pressure = exp(plow+b32*(phigh-plow))
    return pressure/(gas*temp), temp


def forces(x, u, vehicle):
    sin, sqrt = (ca.sin, ca.sqrt) if symbolic(x) else (np.sin, np.sqrt)
    rho, temp = atmosphere(x[0])
    q = .5*rho*x[3]**2
    cl = vehicle.cl0+vehicle.cl_alpha*u[0]
    cd = vehicle.cd0+vehicle.induced_drag*cl**2
    lift, drag = q*vehicle.area*cl, q*vehicle.area*cd
    thrust = u[2]*vehicle.thrust_coefficient*q*vehicle.area
    normal = lift+thrust*sin(u[0])
    heat = vehicle.heat_coefficient*sqrt(rho/vehicle.nose_radius)*x[3]**3
    return lift, drag, thrust, normal, q, heat, x[3]/sqrt(1.4*287.05287*temp)


def derivatives(x, u, vehicle):
    sin, cos, tan = (ca.sin, ca.cos, ca.tan) if symbolic(x) else (np.sin, np.cos, np.tan)
    h, lon, lat, speed, gamma, heading, mass = (x[i] for i in range(7))
    alpha, bank = u[0], u[1]
    lift, drag, thrust, normal, *_ = forces(x, u, vehicle)
    r = vehicle.radius+h
    g, w = vehicle.mu/r**2, vehicle.omega
    sg, cg, sp, cp, sl, cl = sin(gamma), cos(gamma), sin(heading), cos(heading), sin(lat), cos(lat)
    out = [speed*sg, speed*cg*sp/(r*cl), speed*cg*cp/r,
           (thrust*cos(alpha)-drag)/mass-g*sg+w*w*r*cl*(sg*cl-cg*sl*cp),
           normal*cos(bank)/(mass*speed)+(speed/r-g/speed)*cg+2*w*cl*sp
           +w*w*r/speed*cl*(cg*cl+sg*sl*cp),
           normal*sin(bank)/(mass*speed*cg)+speed/r*cg*sp*tan(lat)
           -2*w*(tan(gamma)*cl*cp-sl)+w*w*r/(speed*cg)*sl*cl*sp,
           -thrust/(vehicle.isp*vehicle.g0)]
    return ca.vertcat(*out) if symbolic(x) else np.array(out)


class Discretization:
    """One RK4 graph supplies the NLP, SCP Jacobians, labels and inference."""

    def __init__(self, settings, mission):
        self.settings, self.mission = settings, mission
        self.dt = mission.duration/settings.numerics.intervals
        self.substeps = int(np.ceil(self.dt/settings.numerics.rk4_max_step))
        y, a = ca.SX.sym("y", 7), ca.SX.sym("a", 3)
        scale, offset = ca.DM(STATE_SCALE), ca.DM(mission.offset)
        def rhs(z):
            return derivatives(offset+scale*z, ca.DM(CONTROL_SCALE)*a, settings.vehicle)/scale
        z, step = y, self.dt/self.substeps
        samples = []
        # Retain the substep endpoints used by both optimization methods.
        for _ in range(self.substeps):
            k1 = rhs(z)
            k2 = rhs(z+step*k1/2)
            k3 = rhs(z+step*k2/2)
            k4 = rhs(z+step*k3)
            z = z+step*(k1+2*k2+2*k3+k4)/6
            samples.append(z)
        self.step = ca.Function("transition", [y, a], [z])
        self.linearize = ca.Function("transition_jacobian", [y, a],
                                     [z, ca.jacobian(z, y), ca.jacobian(z, a)])
        # Every RK4 substep endpoint is constrained. The independently replayed
        # trajectory is checked on a finer grid before accepting any solution.
        self.stage_functions = {}
        for nfz in (False, True):
            stage_g = ca.vertcat(*[self.residual_expr(s, a, nfz) for s in [y, *samples]])
            self.stage_functions[nfz] = ca.Function("stage_constraints", [y, a],
                [stage_g, ca.jacobian(stage_g, y), ca.jacobian(stage_g, a)])

    def residual_expr(self, y, a, include_nfz):
        """Dimensionless inequalities g <= 0 for state, path and spherical NFZs."""
        v, lim, num = self.settings.vehicle, self.settings.limits, self.settings.numerics
        x, u = ca.DM(self.mission.offset)+ca.DM(STATE_SCALE)*y, ca.DM(CONTROL_SCALE)*a
        lo, hi, _, _ = bounds(self.settings, self.mission)
        # Small interior reserves cover extrema between two integration samples.
        reserve = np.array([2., 0., 0., .05, 1e-5, 0., 0.])
        lo, hi = lo+reserve, hi-reserve
        margin = num.path_margin
        _, _, _, normal, q, heat, _ = forces(x, u, v)
        # Squared signed normal force avoids an absolute-value derivative at zero.
        result = [(ca.DM(lo)-x)/ca.DM(STATE_SCALE), (x-ca.DM(hi))/ca.DM(STATE_SCALE),
                  (lim.pressure[0]*(1+margin)-q)/lim.pressure[1],
                  q/lim.pressure[1]-(1-margin),
                  heat/lim.heat_max-(1-margin),
                  (normal/(x[6]*v.g0*lim.load_max))**2-(1-margin)**2]
        if include_nfz:
            for zone in self.mission.zones:
                lon, lat = np.radians([zone.longitude_deg, zone.latitude_deg])
                angular = (zone.radius_km*1000+num.nfz_buffer)/v.radius
                dot = ca.sin(x[2])*np.sin(lat)+ca.cos(x[2])*np.cos(lat)*ca.cos(x[1]-lon)
                result.append((dot-np.cos(angular))/(1-np.cos(angular)))
        return ca.vertcat(*result)

    def propagate(self, initial, controls):
        y = self.mission.normalize(initial)
        states = [np.asarray(initial)]
        for u in controls:
            y = np.asarray(self.step(y, np.asarray(u)/CONTROL_SCALE)).ravel()
            states.append(self.mission.physical(y))
        return np.asarray(states)
