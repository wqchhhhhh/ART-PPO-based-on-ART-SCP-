"""One physical model, mission definition and discretization for the whole pipeline."""

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import yaml

MODEL_VERSION = "cruise-art-v1"
STATE_SCALE = np.array([1e4, .1, .1, 1e3, .1, 1., 1e4])
CONTROL_SCALE = np.array([.1, 1., 1.])
FUEL_SCALE = 1000.  # kg; the same scale is used for optimization, RTG and inference


@dataclass(frozen=True)
class Vehicle:
    radius: float = 6371000.
    mu: float = 3.986004418e14
    omega: float = 7.292115e-5
    g0: float = 9.80665
    area: float = 60.
    dry_mass: float = 12000.
    cl0: float = -.010
    cl_alpha: float = 1.55
    cd0: float = .016
    induced_drag: float = .75
    thrust_coefficient: float = .040
    isp: float = 1500.
    nose_radius: float = .05
    heat_coefficient: float = 1.7415e-4


@dataclass(frozen=True)
class Limits:
    altitude: tuple = (25000., 35000.)
    speed: tuple = (1850., 2350.)
    gamma_deg: tuple = (-4., 4.)
    alpha_deg: tuple = (-1.5, 8.)
    bank_deg: tuple = (-60., 60.)
    throttle: tuple = (.05, 1.)
    pressure: tuple = (18000., 80000.)
    heat_max: float = 1e6
    load_max: float = 2.5


@dataclass(frozen=True)
class Zone:
    longitude_deg: float
    latitude_deg: float
    radius_km: float


@dataclass(frozen=True)
class Mission:
    # State: altitude, longitude, latitude, speed, gamma, heading, mass.
    # SI units; heading is clockwise from north.
    initial: tuple
    target: tuple  # first five components are fixed; mass and heading are free
    duration: float
    zones: tuple = ()
    scenario_id: int = 0

    @property
    def offset(self):
        return np.array([0., self.initial[1], self.initial[2], 0., 0., 0., 0.])

    def normalize(self, states):
        return (np.asarray(states) - self.offset) / STATE_SCALE

    def physical(self, states):
        return np.asarray(states) * STATE_SCALE + self.offset

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        value["initial"] = tuple(value["initial"])
        value["target"] = tuple(value["target"])
        value["zones"] = tuple(Zone(**z) for z in value.get("zones", []))
        return cls(**value)


@dataclass(frozen=True)
class Numerics:
    intervals: int = 100
    rk4_max_step: float = 2.
    validation_step: float = 1.
    nfz_buffer: float = 200.
    path_margin: float = .001
    expert_iterations: int = 500
    scp_iterations: int = 40
    trust_initial: float = 1.
    trust_min: float = 1e-4
    penalty: float = 100.
    defect_tolerance: float = 1e-7
    cost_tolerance: float = 1e-5
    step_tolerance: float = 1e-3


@dataclass(frozen=True)
class Settings:
    vehicle: Vehicle = field(default_factory=Vehicle)
    limits: Limits = field(default_factory=Limits)
    numerics: Numerics = field(default_factory=Numerics)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        return cls(Vehicle(**value["vehicle"]), Limits(**value["limits"]),
                   Numerics(**value["numerics"]))


def load_problem(path):
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    settings = Settings(Vehicle(**raw.get("vehicle", {})),
                        Limits(**raw.get("limits", {})),
                        Numerics(**raw.get("numerics", {})))
    spec = raw["mission"]
    from cruise.geometry import bearing
    lon0, lat0 = np.radians(spec["start_deg"])
    lonf, latf = np.radians(spec["target_deg"])
    initial = (spec["altitude_initial"], lon0, lat0, spec["speed_initial"],
               np.radians(spec.get("gamma_initial_deg", 0.)),
               np.radians(spec["heading_initial_deg"]) if "heading_initial_deg" in spec
               else bearing(lon0, lat0, lonf, latf), spec["mass_initial"])
    target = (spec["altitude_final"], lonf, latf, spec["speed_final"],
              np.radians(spec.get("gamma_final_deg", 0.)))
    mission = Mission(initial, target, spec["duration"],
                      tuple(Zone(**z) for z in spec.get("zones", [])))
    if settings.numerics.intervals < 2 or mission.duration <= 0:
        raise ValueError("Positive duration and at least two control intervals are required")
    return settings, mission


def bounds(settings, mission):
    v, lim = settings.vehicle, settings.limits
    lo = np.array([lim.altitude[0], min(mission.initial[1], mission.target[1])-.2,
                   min(mission.initial[2], mission.target[2])-.2, lim.speed[0],
                   np.radians(lim.gamma_deg[0]), -np.pi, v.dry_mass])
    hi = np.array([lim.altitude[1], max(mission.initial[1], mission.target[1])+.2,
                   max(mission.initial[2], mission.target[2])+.2, lim.speed[1],
                   np.radians(lim.gamma_deg[1]), np.pi, mission.initial[6]])
    ulo = np.array([np.radians(lim.alpha_deg[0]), np.radians(lim.bank_deg[0]), lim.throttle[0]])
    uhi = np.array([np.radians(lim.alpha_deg[1]), np.radians(lim.bank_deg[1]), lim.throttle[1]])
    return lo, hi, ulo, uhi
