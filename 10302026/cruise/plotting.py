"""Physical-unit plots for saved, independently validated cruise solutions."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cruise.config import Mission, Settings
from cruise.dynamics import forces


def plot_solution(path, output):
    with np.load(path, allow_pickle=False) as data:
        x, u = data["states"], data["controls"]
        mission = Mission.from_dict(json.loads(str(data["mission"])))
        settings = Settings.from_dict(json.loads(str(data["settings"])))
    t = np.linspace(0., mission.duration, len(x))
    tu = t[:-1]
    _, _, _, normal, q, heat, mach = forces(x[:-1].T, u.T, settings.vehicle)
    fig, axes = plt.subplots(4, 3, figsize=(14, 14), constrained_layout=True)
    ax = axes.ravel()
    ax[0].plot(np.degrees(x[:, 1]), np.degrees(x[:, 2]), color="#096B72")
    for z in mission.zones:
        theta = np.linspace(0., 2*np.pi, 160)
        radius = np.degrees(z.radius_km*1000/settings.vehicle.radius)
        ax[0].fill(z.longitude_deg+radius*np.cos(theta)/np.cos(np.radians(z.latitude_deg)),
                   z.latitude_deg+radius*np.sin(theta), color="#DD775B", alpha=.35)
    ax[0].set(xlabel="Longitude [deg]", ylabel="Latitude [deg]", title="Ground track")
    curves = [(1, t, x[:, 0]/1000, "Altitude [km]"), (2, t, x[:, 3], "Speed [m/s]"),
              (3, t, np.degrees(x[:, 4]), "Flight path angle [deg]"),
              (4, t, np.degrees(x[:, 5]), "Heading [deg]"), (5, t, x[:, 6], "Mass [kg]"),
              (6, tu, np.degrees(u[:, 0]), "Angle of attack [deg]"),
              (7, tu, np.degrees(u[:, 1]), "Bank [deg]"), (8, tu, u[:, 2], "Throttle"),
              (9, tu, q/1000, "Dynamic pressure [kPa]"), (10, tu, heat/1000, "Heat flux [kW/m2]"),
              (11, tu, np.abs(normal)/(x[:-1, 6]*settings.vehicle.g0), "Normal load [g]")]
    for index, time, value, label in curves:
        if index in (6, 7, 8):
            ax[index].step(np.r_[time, t[-1]], np.r_[value, value[-1]], where="post", color="#096B72")
        else:
            ax[index].plot(time, value, color="#096B72")
        ax[index].set(xlabel="Time [s]", ylabel=label)
    for i in (9,):
        for value in settings.limits.pressure:
            ax[i].axhline(value/1000, linestyle="--", color="#B14D3A", linewidth=1)
    ax[10].axhline(settings.limits.heat_max/1000, linestyle="--", color="#B14D3A", linewidth=1)
    ax[11].axhline(settings.limits.load_max, linestyle="--", color="#B14D3A", linewidth=1)
    for axis in ax:
        axis.grid(alpha=.22)
    fig.suptitle("Air-breathing cruise: optimized states and held controls", fontsize=16)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170)
    plt.close(fig)
