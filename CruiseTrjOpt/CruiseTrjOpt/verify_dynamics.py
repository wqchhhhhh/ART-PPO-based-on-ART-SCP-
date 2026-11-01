"""Independent continuous dynamics and path-constraint verification."""

import numpy as np
import matplotlib.pyplot as plt

import config as cfg
from continuous import validate_solution, require_valid_solution


def forward_integrate_trajectory(sol, method="DOP853", rtol=1e-10, atol=1e-11):
    report = validate_solution(sol,method=method,rtol=rtol,atol=atol)
    data = sol["_verification"]
    state_names = ["Altitude h","Longitude theta","Latitude phi","Velocity V",
                   "Flight path angle gamma","Heading psi","Mass m"]
    units = ["m","deg","deg","m/s","deg","deg","kg"]
    metrics = {}
    for j,name in enumerate(state_names):
        error = data["errors"][:,j]
        values = sol["states"][:,j].copy()
        if j==0:
            values -= cfg.R_E
        factor = 180./np.pi if j in (1,2,4,5) else 1.
        metrics[name] = {"unit":units[j],"max_error":float(error.max()*factor),
                         "mean_error":float(error.mean()*factor),
                         "rmse":float(np.sqrt(np.mean(error**2))*factor),
                         "rel_error_pct":float(100*error.mean()/max(np.mean(abs(values)),1e-6))}
    data["metrics"] = metrics
    data["validation"] = report
    return data


def print_verification_report(sim_data):
    validation = sim_data["validation"]
    print("\nContinuous trajectory verification: " + ("PASS" if validation["passed"] else "FAIL"))
    print(f"{validation['integration_method']}; max step={validation['max_step_s']} s; "
          f"{validation['grid_points']} grid points; refined local extrema")
    for name,metric in sim_data.get("metrics",{}).items():
        print(f"  {name:<25} max error={metric['max_error']:.6g} {metric['unit']}")
    for name,check in validation["checks"].items():
        print(f"  {name:<25} [{check['min']:.8g}, {check['max']:.8g}] "
              f"{'PASS' if check['passed'] else 'FAIL'}")
    for name,check in validation["nfz_checks"].items():
        print(f"  {name:<25} min clearance={check['min_clearance_m']:.6f} m "
              f"{'PASS' if check['passed'] else 'FAIL'}")
    if validation["failures"]:
        print("Failed checks: " + ", ".join(validation["failures"]))
    return validation["passed"]


def plot_verification_comparison(sim_data, save_path="results/dynamics_verification.png"):
    t = sim_data["time"]
    x = sim_data["states_opt"]
    y = sim_data["states_sim"]
    error = sim_data["errors"]
    fig,axes = plt.subplots(3,2,figsize=(12,10),layout="constrained")
    for ax,j,label,scale,offset in ((axes[0,0],0,"Altitude [km]",.001,cfg.R_E),
                                    (axes[0,1],3,"Speed [m/s]",1.,0.),
                                    (axes[1,0],4,"Flight path angle [deg]",180/np.pi,0.),
                                    (axes[1,1],5,"Heading [deg]",180/np.pi,0.),
                                    (axes[2,0],6,"Mass [kg]",1.,0.)):
        ax.plot(t,(x[:,j]-offset)*scale,"b.",label="Shooting knots")
        ax.plot(t,(y[:,j]-offset)*scale,"r-",label="DOP853 replay")
        ax.set(xlabel="Time [s]",ylabel=label)
        ax.legend()
    ax = axes[2,1]
    horizontal = cfg.R_E*np.hypot(np.cos(x[:,2])*error[:,1],error[:,2])
    ax.plot(t,error[:,0],label="Altitude error [m]")
    ax.plot(t,horizontal,label="Horizontal error [m]")
    ax.set(xlabel="Time [s]",ylabel="Independent integration error [m]")
    ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=.3)
    fig.suptitle("Independent dynamics verification: "+("PASS" if sim_data["passed"] else "FAIL"))
    fig.savefig(save_path,dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    from optimizer import TrajectoryOptimizer
    sol = TrajectoryOptimizer().solve(print_level=0)
    if not sol["success"]:
        raise SystemExit("Optimization/validation failed: "+str(sol.get("validation")))
    sim_data = forward_integrate_trajectory(sol)
    print_verification_report(sim_data)
    require_valid_solution(sol)
    plot_verification_comparison(sim_data)
