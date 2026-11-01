"""Plots of independently propagated trajectories and their actual constraints."""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from constraints import Constraints
from continuous import sample_solution
from dynamics import compute_dynamics


def setup_plot_style():
    plt.rcParams.update({"font.family":"sans-serif","font.size":10,"axes.grid":True,
                         "grid.alpha":.3,"lines.linewidth":1.6,"figure.dpi":160})


def _dense(sol):
    time = sol.get("_verification",{}).get("dense_time",np.linspace(0.,sol["tf"],2001))
    states,controls = sample_solution(sol,time)
    _,aux = compute_dynamics(states.T,controls.T)
    return time,states,controls,aux


def _save(fig,path):
    fig.savefig(path,dpi=180,bbox_inches="tight")
    plt.close(fig)


def plot_ground_track(sol,initial_guess=None,save_path="results/ground_track.png"):
    time,x,u,aux = _dense(sol)
    fig,ax = plt.subplots(figsize=(10,6),layout="constrained")
    for nfz in sol["nfzs"]:
        lon,lat = Constraints.no_fly_zone_boundary(nfz)
        ax.fill(lon,lat,color="crimson",alpha=.2,label=f"{nfz['name']} ({nfz['radius_km']:.0f} km)")
        ax.plot(lon,lat,color="crimson",ls="--",lw=1.)
    if initial_guess is not None:
        ax.plot(np.degrees(initial_guess["states"][:,1]),np.degrees(initial_guess["states"][:,2]),
                color="gray",ls=":",label="Geometric initial guess")
    ax.plot(np.degrees(x[:,1]),np.degrees(x[:,2]),color="navy",label="Validated trajectory")
    ax.scatter(np.degrees(sol["states"][:,1]),np.degrees(sol["states"][:,2]),s=5,color="navy",alpha=.3)
    ax.plot(np.degrees(x[0,1]),np.degrees(x[0,2]),"go",label="Start")
    ax.plot(np.degrees(x[-1,1]),np.degrees(x[-1,2]),"m*",ms=11,label="Target")
    ax.set(xlabel="Longitude [deg E]",ylabel="Latitude [deg N]",title="Continuous ground track and geodesic no-fly zones")
    ax.set_aspect(1./np.cos(np.mean(x[:,2])))
    ax.legend(fontsize=8)
    _save(fig,save_path)


def plot_state_histories(sol,save_path="results/state_histories.png"):
    t,x,u,a = _dense(sol)
    fig,axes = plt.subplots(3,2,figsize=(12,10),layout="constrained")
    values = [(x[:,0]-cfg.R_E)/1000.,x[:,3],np.degrees(x[:,4]),np.degrees(x[:,5]),x[:,6],a["Ma"]]
    labels = ["Altitude [km]","Speed [m/s]","Flight path angle [deg]","Heading [deg]","Mass [kg]","Mach number"]
    bounds = [(cfg.ALT_MIN/1000.,cfg.ALT_MAX/1000.),(cfg.VEL_MIN,cfg.VEL_MAX),
              tuple(np.degrees([cfg.GAMMA_MIN,cfg.GAMMA_MAX])),tuple(np.degrees([cfg.PSI_MIN,cfg.PSI_MAX])),
              (cfg.MASS_MIN,sol["states"][0,6]),None]
    for ax,value,label,bound in zip(axes.flat,values,labels,bounds):
        ax.plot(t,value)
        ax.set(xlabel="Time [s]",ylabel=label)
        if bound is not None:
            for v in bound:
                ax.axhline(v,color="red",ls="--",lw=1.)
    fig.suptitle("States from independent continuous propagation")
    _save(fig,save_path)


def plot_control_histories(sol,save_path="results/control_histories.png"):
    t,x,u,a = _dense(sol)
    fig,axes = plt.subplots(3,1,figsize=(10,8),layout="constrained")
    values = [np.degrees(u[:,0]),np.degrees(u[:,1]),u[:,2]]
    labels = ["Angle of attack [deg]","Bank angle [deg]","Throttle"]
    bounds = [np.degrees([cfg.ALPHA_MIN,cfg.ALPHA_MAX]),np.degrees([cfg.SIGMA_MIN,cfg.SIGMA_MAX]),[cfg.ETA_MIN,cfg.ETA_MAX]]
    for ax,value,label,bound in zip(axes,values,labels,bounds):
        ax.plot(t,value)
        ax.set(xlabel="Time [s]",ylabel=label)
        for b in bound:
            ax.axhline(b,color="red",ls="--",lw=1.)
    fig.suptitle("Piecewise-linear optimization controls")
    _save(fig,save_path)


def plot_constraints_and_nfz(sol,save_path="results/constraints_and_nfz.png"):
    t,x,u,a = _dense(sol)
    fig,axes = plt.subplots(2,2,figsize=(12,8),layout="constrained")
    nfzs = Constraints.no_fly_zone_distances_km(np.degrees(x[:,1]),np.degrees(x[:,2]),sol["nfzs"])
    for name,data in nfzs.items():
        axes[0,0].plot(t,data["clearance_km"],label=name)
    axes[0,0].axhline(0.,color="red",ls="--")
    axes[0,0].set(ylabel="NFZ clearance [km]",title=f"Minimum clearance: {sol['validation']['min_clearance_m']:.3f} m" if nfzs else "No no-fly zones")
    if nfzs:
        axes[0,0].legend()
    for ax,key,scale,label,bounds in ((axes[0,1],"q",.001,"Dynamic pressure [kPa]",(cfg.Q_MIN/1000.,cfg.Q_MAX/1000.)),
                                      (axes[1,0],"n_load",1.,"Normal load [g]",(cfg.N_MAX,)),
                                      (axes[1,1],"q_heat",.001,"Heat flux [kW/m2]",(cfg.Q_HEAT_MAX/1000.,))):
        ax.plot(t,a[key]*scale)
        ax.set(ylabel=label)
        for value in bounds:
            ax.axhline(value,color="red",ls="--")
    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
    fig.suptitle("Continuous path constraints: "+("PASS" if sol["validation"]["passed"] else "FAIL"))
    _save(fig,save_path)


def plot_3d_trajectory(sol,save_path="results/trajectory_3d.png"):
    t,x,u,a = _dense(sol)
    fig = plt.figure(figsize=(11,7),layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot(np.degrees(x[:,1]),np.degrees(x[:,2]),(x[:,0]-cfg.R_E)/1000.,color="navy")
    for nfz in sol["nfzs"]:
        lon,lat = Constraints.no_fly_zone_boundary(nfz)
        for h in (cfg.ALT_MIN/1000.,cfg.ALT_MAX/1000.):
            ax.plot(lon,lat,np.full_like(lon,h),color="crimson",alpha=.6)
        for j in range(0,len(lon),15):
            ax.plot([lon[j],lon[j]],[lat[j],lat[j]],[cfg.ALT_MIN/1000.,cfg.ALT_MAX/1000.],color="crimson",alpha=.2)
    ax.set(xlabel="Longitude [deg E]",ylabel="Latitude [deg N]",zlabel="Altitude [km]",title="Continuous 3D trajectory")
    ax.view_init(elev=27,azim=-65)
    _save(fig,save_path)


def generate_all_plots(sol,initial_guess=None,output_dir="results"):
    setup_plot_style()
    destination = Path(output_dir)
    plot_ground_track(sol,initial_guess,str(destination/"ground_track.png"))
    plot_state_histories(sol,str(destination/"state_histories.png"))
    plot_control_histories(sol,str(destination/"control_histories.png"))
    plot_constraints_and_nfz(sol,str(destination/"constraints_and_nfz.png"))
    plot_3d_trajectory(sol,str(destination/"trajectory_3d.png"))
