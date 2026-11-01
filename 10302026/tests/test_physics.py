import numpy as np

from cruise.config import CONTROL_SCALE, Mission, Numerics, Settings, Vehicle
from cruise.dynamics import Discretization, atmosphere, derivatives, forces


def cartesian_reference(x, u, v):
    h, lon, lat, speed, gamma, psi, mass = x
    er = np.array([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)])
    east = np.array([-np.sin(lon), np.cos(lon), 0.])
    north = np.cross(er, east)
    horizontal = np.sin(psi)*east+np.cos(psi)*north
    direction = np.cos(gamma)*horizontal+np.sin(gamma)*er
    gamma_direction = -np.sin(gamma)*horizontal+np.cos(gamma)*er
    heading_direction = np.cos(psi)*east-np.sin(psi)*north
    lift, drag, thrust, normal, *_ = forces(x, u, v)
    force = (thrust*np.cos(u[0])-drag)*direction+normal*(np.cos(u[1])*gamma_direction+np.sin(u[1])*heading_direction)
    omega = np.array([0., 0., v.omega])
    radius = v.radius+h
    acceleration = (force/mass-v.mu/radius**2*er-2*np.cross(omega, speed*direction)
                    -np.cross(omega, np.cross(omega, radius*er)))
    return np.array([speed*np.sin(gamma), speed*np.cos(gamma)*np.sin(psi)/(radius*np.cos(lat)),
                     speed*np.cos(gamma)*np.cos(psi)/radius, acceleration@direction,
                     acceleration@gamma_direction/speed+speed*np.cos(gamma)/radius,
                     acceleration@heading_direction/(speed*np.cos(gamma))
                     +speed*np.cos(gamma)*np.sin(psi)*np.tan(lat)/radius,
                     -thrust/(v.isp*v.g0)])


def test_rotating_earth_dynamics_against_cartesian_force_balance():
    rng, vehicle = np.random.default_rng(7), Vehicle()
    for _ in range(64):
        x = np.array([rng.uniform(25000, 35000), rng.uniform(-2, 2), rng.uniform(-.8, .8),
                      rng.uniform(1850, 2350), rng.uniform(-.06, .06), rng.uniform(-2.5, 2.5), rng.uniform(12000, 21000)])
        u = np.array([rng.uniform(0, .13), rng.uniform(-.8, .8), rng.uniform(.05, 1)])
        np.testing.assert_allclose(derivatives(x, u, vehicle), cartesian_reference(x, u, vehicle), atol=2e-12, rtol=2e-12)


def test_us76_density_at_30km():
    rho, temp = atmosphere(30000.)
    assert abs(rho/.0184101-1) < 1e-4
    assert abs(temp-226.509) < .01


def test_transition_sensitivities_against_finite_differences():
    x = (30000., 1.8, .35, 2110., 0., 1.1, 20000.)
    mission = Mission(x, x[:5], 20.)
    settings = Settings(numerics=Numerics(intervals=2))
    d = Discretization(settings, mission)
    y, a = mission.normalize(x), np.array([.06, .1, .6])/CONTROL_SCALE
    _, jx, ju = d.linearize(y, a)
    both = np.r_[y, a]
    fd = []
    for i in range(10):
        eps = np.eye(10)[i]*1e-6
        p, q = both+eps, both-eps
        fd.append((np.asarray(d.step(p[:7], p[7:])).ravel()-np.asarray(d.step(q[:7], q[7:])).ravel())/2e-6)
    np.testing.assert_allclose(np.c_[np.asarray(jx), np.asarray(ju)], np.array(fd).T, atol=1e-7, rtol=2e-5)
