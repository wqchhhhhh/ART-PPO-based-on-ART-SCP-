"""Run four missions and publish continuous-trajectory acceptance results."""

import json
from pathlib import Path
import sys
import time
from datetime import datetime,timezone,timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config as cfg
from optimizer import TrajectoryOptimizer
from constraints import Constraints
from continuous import sample_solution,save_solution

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")


SCENARIOS = [
    {
        "id": "Scenario-1",
        "name": "Baseline (3 NFZs, Nominal Corridor)",
        "lon_0_deg": 105.0, "lat_0_deg": 20.0,
        "lon_f_deg": 143.0, "lat_f_deg": 32.0,
        "color": "blue",
        "no_fly_zones": [
            {"name": "NFZ-1", "lon_deg": 116.0, "lat_deg": 23.5, "radius_km": 180.0},
            {"name": "NFZ-2", "lon_deg": 126.0, "lat_deg": 27.0, "radius_km": 200.0},
            {"name": "NFZ-3", "lon_deg": 134.0, "lat_deg": 29.5, "radius_km": 180.0},
        ]
    },
    {
        "id": "Scenario-2",
        "name": "South Shift (South Maritime Corridor)",
        "lon_0_deg": 102.0, "lat_0_deg": 16.0,
        "lon_f_deg": 140.0, "lat_f_deg": 28.0,
        "color": "green",
        "no_fly_zones": [
            {"name": "NFZ-S1", "lon_deg": 113.0, "lat_deg": 19.5, "radius_km": 180.0},
            {"name": "NFZ-S2", "lon_deg": 123.0, "lat_deg": 23.0, "radius_km": 200.0},
            {"name": "NFZ-S3", "lon_deg": 131.0, "lat_deg": 25.5, "radius_km": 180.0},
        ]
    },
    {
        "id": "Scenario-3",
        "name": "Expanded NFZ Bottleneck (Enlarged Radii 230-250km)",
        "lon_0_deg": 105.0, "lat_0_deg": 20.0,
        "lon_f_deg": 143.0, "lat_f_deg": 32.0,
        "color": "crimson",
        "no_fly_zones": [
            {"name": "NFZ-E1", "lon_deg": 116.0, "lat_deg": 23.5, "radius_km": 230.0},
            {"name": "NFZ-E2", "lon_deg": 126.0, "lat_deg": 27.0, "radius_km": 250.0},
            {"name": "NFZ-E3", "lon_deg": 134.0, "lat_deg": 29.5, "radius_km": 230.0},
        ]
    },
    {
        "id": "Scenario-4",
        "name": "Zigzag Route (4 Staggered NFZs, North Ascending)",
        "lon_0_deg": 104.0, "lat_0_deg": 24.0,
        "lon_f_deg": 145.0, "lat_f_deg": 36.0,
        "color": "purple",
        "no_fly_zones": [
            {"name": "NFZ-Z1", "lon_deg": 113.0, "lat_deg": 26.5, "radius_km": 160.0},
            {"name": "NFZ-Z2", "lon_deg": 122.0, "lat_deg": 29.5, "radius_km": 180.0},
            {"name": "NFZ-Z3", "lon_deg": 130.0, "lat_deg": 33.0, "radius_km": 170.0},
            {"name": "NFZ-Z4", "lon_deg": 138.0, "lat_deg": 34.5, "radius_km": 160.0},
        ]
    },
]


def run_scenario_experiments():
    rows=[]
    solutions={}
    reports=[]
    for index,scenario in enumerate(SCENARIOS,1):
        print(f"Running {scenario['id']}: {scenario['name']}",flush=True)
        sol=TrajectoryOptimizer(scenario_config=scenario).solve(max_iter=500,tol=1e-8,print_level=0)
        solutions[scenario["id"]]=sol
        report=sol["validation"]
        reports.append({"id":scenario["id"],"scenario":scenario,"accepted":sol["success"],
                        "solver_status":sol["return_status"],"solve_time_s":sol["solve_time"],
                        "flight_time_s":sol["tf"],"fuel_kg":sol["fuel_consumed_kg"],
                        "refinement_history":sol.get("refinement_history",[]),"validation":report})
        row={"Scenario ID":scenario["id"],"Description":scenario["name"],"Status":sol["return_status"],
             "Continuous PASS":sol["success"],"Wall Time [s]":sol["solve_time"],
             "Flight Time [s]":sol["tf"],"Fuel Burned [kg]":sol["fuel_consumed_kg"]}
        if sol["success"]:
            checks=report["checks"]
            row.update({"Min Clearance [m]":report["min_clearance_m"],
                        "Altitude Min [m]":checks["altitude_m"]["min"],"Altitude Max [m]":checks["altitude_m"]["max"],
                        "Speed Min [m/s]":checks["velocity_mps"]["min"],"Speed Max [m/s]":checks["velocity_mps"]["max"],
                        "Throttle Min":checks["throttle"]["min"],"Throttle Max":checks["throttle"]["max"],
                        "Max Load [g]":checks["normal_load"]["max"],"Max Heat [kW/m2]":checks["heat_flux_wm2"]["max"]/1000.,
                        "Max Altitude Error [m]":report["dynamics_max_abs_error_si"][0],
                        "Max Velocity Error [m/s]":report["dynamics_max_abs_error_si"][3],
                        "Max Mass Error [kg]":report["dynamics_max_abs_error_si"][6]})
            save_solution(sol,f"results/scenario_{index}_solution.npz")
            from main import save_trajectory_csv
            save_trajectory_csv(sol,f"results/scenario_{index}_continuous.csv")
        rows.append(row)
        print(f"  accepted={sol['success']}; fuel={sol['fuel_consumed_kg']:.3f} kg; "
              f"tf={sol['tf']:.3f} s; clearance={report.get('min_clearance_m')} m; "
              f"elapsed={sol['solve_time']:.2f} s; failures={report.get('failures')}",flush=True)
    table=pd.DataFrame(rows)
    table.to_csv("results/scenario_comparison.csv",index=False)
    overall=all(s["success"] for s in solutions.values())
    result={"model_version":cfg.MODEL_VERSION,"timestamp":datetime.now(timezone(timedelta(hours=8))).isoformat(),
            "all_passed":overall,"scenarios":reports}
    Path("results/scenario_validation.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(table.to_string(index=False))
    if not overall:
        raise RuntimeError("One or more scenarios failed; see results/scenario_validation.json")
    plot_scenario_comparison(solutions)
    from main import publish_baseline
    publish_baseline(solutions["Scenario-1"])
    return table,solutions


def plot_scenario_comparison(solutions,save_path="results/scenario_comparison.png"):
    fig=plt.figure(figsize=(13,10),layout="constrained")
    grid=fig.add_gridspec(2,2)
    track=fig.add_subplot(grid[0,:])
    altitude=fig.add_subplot(grid[1,0])
    bank=fig.add_subplot(grid[1,1])
    for scenario in SCENARIOS:
        sol=solutions[scenario["id"]]
        color=scenario["color"]
        for nfz in sol["nfzs"]:
            lon,lat=Constraints.no_fly_zone_boundary(nfz)
            track.fill(lon,lat,color=color,alpha=.1)
        t=np.linspace(0.,sol["tf"],1001)
        x,u=sample_solution(sol,t)
        track.plot(np.degrees(x[:,1]),np.degrees(x[:,2]),color=color,label=scenario["id"])
        track.plot(np.degrees(x[0,1]),np.degrees(x[0,2]),"o",color=color)
        track.plot(np.degrees(x[-1,1]),np.degrees(x[-1,2]),"*",color=color)
        altitude.plot(t,(x[:,0]-cfg.R_E)/1000.,color=color,label=scenario["id"])
        bank.plot(t,np.degrees(u[:,1]),color=color,label=scenario["id"])
    track.set(xlabel="Longitude [deg E]",ylabel="Latitude [deg N]",title="Validated continuous trajectories")
    altitude.set(xlabel="Time [s]",ylabel="Altitude [km]")
    bank.set(xlabel="Time [s]",ylabel="Bank angle [deg]")
    for h in [cfg.ALT_MIN/1000.,cfg.ALT_MAX/1000.]:
        altitude.axhline(h,color="red",ls="--",alpha=.5)
    for ax in [track,altitude,bank]:
        ax.grid(alpha=.3)
        ax.legend()
    fig.savefig(save_path,dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    run_scenario_experiments()
