"""Run the baseline mission and publish continuously validated results."""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

import config as cfg
from optimizer import TrajectoryOptimizer
from initial_guess import build_initial_guess
from continuous import sample_solution, describe_trajectory, require_valid_solution, save_solution
from visualization import generate_all_plots
from verify_dynamics import print_verification_report, plot_verification_comparison

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")


def print_mission_banner():
    print("AHV fuel-optimal cruise: LGL initialization + continuous multiple shooting")
    print(f"Start ({cfg.LON_0_DEG}, {cfg.LAT_0_DEG}); target ({cfg.LON_F_DEG}, {cfg.LAT_F_DEG}); "
          f"{len(cfg.NO_FLY_ZONES)} NFZs; tf in [{cfg.TF_MIN}, {cfg.TF_MAX}] s")


def print_summary_report(sol):
    print(f"\nSolver: {sol['return_status']}; accepted: {sol['success']}; elapsed: {sol['solve_time']:.2f} s")
    print(f"Flight time: {sol['tf']:.3f} s; fuel: {sol['fuel_consumed_kg']:.3f} kg")
    if "_verification" in sol:
        return print_verification_report(sol["_verification"])
    print("Continuous checks: NOT VERIFIED; "+str(sol.get("validation",{}).get("failures",[])))
    return False


def save_trajectory_csv(sol, filepath="results/optimal_trajectory.csv"):
    require_valid_solution(sol)
    time = np.unique(np.r_[sol["time"],np.linspace(0.,sol["tf"],1001)])
    states,controls = sample_solution(sol,time)
    data = describe_trajectory(time,states,controls,sol["nfzs"],sol["scenario"])
    columns = {"time_s":time,"altitude_m":data["altitude"],"longitude_deg":data["longitude_deg"],
               "latitude_deg":data["latitude_deg"],"velocity_mps":data["velocity"],
               "mach_number":data["mach"],"gamma_deg":data["gamma_deg"],"psi_deg":data["psi_deg"],
               "mass_kg":data["mass"],"alpha_deg":data["alpha_deg"],"bank_angle_deg":data["sigma_deg"],
               "throttle":data["throttle"],"dynamic_pressure_kpa":data["dynamic_pressure_kpa"],
               "load_factor_g":data["load_factor"],"heat_flux_kw_m2":data["heat_flux_kw"]}
    pd.DataFrame(columns).to_csv(filepath,index=False)


def publish_baseline(sol):
    require_valid_solution(sol)
    save_trajectory_csv(sol)
    save_solution(sol,"results/optimal_solution.npz")
    Path("results/baseline_validation.json").write_text(json.dumps(sol["validation"],indent=2),encoding="utf-8")
    generate_all_plots(sol,build_initial_guess(cfg.N_NODES),output_dir="results")
    plot_verification_comparison(sol["_verification"])


def main():
    print_mission_banner()
    sol = TrajectoryOptimizer().solve(print_level=0)
    print_summary_report(sol)
    require_valid_solution(sol)
    publish_baseline(sol)
    print("Validated baseline results saved in results/.")


if __name__ == "__main__":
    main()
