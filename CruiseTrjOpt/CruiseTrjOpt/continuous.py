"""Shared physical trajectory, independent propagation and numerical acceptance."""

import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
from scipy.signal import find_peaks

import config as cfg
from constraints import Constraints
from dynamics import compute_dynamics, compute_dynamics_reference


def control_history(sol):
    """The same continuous, bound-preserving control law is used everywhere."""
    return interp1d(sol["time"],sol["controls"],axis=0,kind="linear",
                    bounds_error=True,assume_sorted=True)


def propagate_solution(sol, method="DOP853", rtol=1e-10, atol=1e-11, max_step=None):
    control = control_history(sol)
    result = solve_ivp(lambda t,x: compute_dynamics(x,control(t))[0],
                       (0.,float(sol["tf"])),sol["states"][0],method=method,
                       rtol=rtol,atol=atol,max_step=max_step or cfg.VALIDATION_MAX_STEP,
                       dense_output=True)
    if not result.success:
        raise RuntimeError("Forward integration failed: " + result.message)
    return result, control


def sample_solution(sol, time):
    """Sample propagated states and the exact optimization control law."""
    if "_dense_solution" not in sol:
        propagation, _ = propagate_solution(sol)
        sol["_dense_solution"] = propagation.sol
    t = np.asarray(time,dtype=float)
    return sol["_dense_solution"](t).T, control_history(sol)(t)


def describe_trajectory(time, states, controls, nfzs, scenario):
    """Create the public result fields from physical SI states and controls."""
    time = np.asarray(time,dtype=float)
    states = np.asarray(states,dtype=float)
    controls = np.asarray(controls,dtype=float)
    _, aux = compute_dynamics(states.T,controls.T)
    return {"time":time,"tf":float(time[-1]),"tau":2*time/time[-1]-1,
            "states":states,"controls":controls,"nfzs":nfzs,"scenario":scenario,
            "altitude":states[:,0]-cfg.R_E,"longitude_deg":np.degrees(states[:,1]),
            "latitude_deg":np.degrees(states[:,2]),"velocity":states[:,3],
            "gamma_deg":np.degrees(states[:,4]),"psi_deg":np.degrees(states[:,5]),
            "mass":states[:,6],"alpha_deg":np.degrees(controls[:,0]),
            "sigma_deg":np.degrees(controls[:,1]),"throttle":controls[:,2],
            "dynamic_pressure_kpa":aux["q"]/1000.,"load_factor":aux["n_load"],
            "heat_flux_kw":aux["q_heat"]/1000.,"mach":aux["Ma"],
            "fuel_consumed_kg":float(states[0,6]-states[-1,6])}


def _extrema(time, values, function):
    """Refine every sampled local extremum with the dense ODE interpolant."""
    minimum, maximum = float(np.min(values)),float(np.max(values))
    prominence = max(1e-10,float(np.ptp(values))*1e-10)
    for sign in (1.,-1.):
        candidates = find_peaks(-sign*values,prominence=prominence)[0]
        for i in candidates:
            result = minimize_scalar(lambda t: sign*function(t),
                                     bounds=(float(time[i-1]),float(time[i+1])),
                                     method="bounded",options={"xatol":1e-6})
            value = float(sign*result.fun)
            minimum = min(minimum,value)
            maximum = max(maximum,value)
    return minimum,maximum


def validate_solution(sol, max_step=None, method="DOP853", rtol=1e-10, atol=1e-11):
    """Numerical continuous-trajectory checks with explicit physical tolerances.

    A <=1 s grid including every control knot brackets local extrema. Their
    values are refined using the dense independent ODE solution. This is a
    numerical verification with documented tolerances.
    """
    step = max_step or cfg.VALIDATION_MAX_STEP
    propagation, control = propagate_solution(sol,method,rtol,atol,step)
    dense = propagation.sol
    time = np.unique(np.r_[np.linspace(0.,sol["tf"],int(np.ceil(sol["tf"]/step))+1),sol["time"]])
    states = dense(time).T
    controls = control(time)
    _, aux = compute_dynamics(states.T,controls.T)
    scenario = sol.get("scenario",{})
    m0 = scenario.get("m_init",cfg.M_INIT)
    checks = {}

    def check(name,values,function,lower,upper,tolerance):
        low,high = _extrema(time,np.asarray(values),function)
        checks[name] = {"min":low,"max":high,"lower":float(lower),"upper":float(upper),
                        "tolerance":float(tolerance),
                        "passed":bool(np.isfinite([low,high]).all()
                                      and low>=lower-tolerance and high<=upper+tolerance)}

    state_specs = [("altitude_m",0,cfg.R_E,cfg.ALT_MIN,cfg.ALT_MAX,.05),
                   ("velocity_mps",3,0.,cfg.VEL_MIN,cfg.VEL_MAX,.005),
                   ("gamma_rad",4,0.,cfg.GAMMA_MIN,cfg.GAMMA_MAX,1e-7),
                   ("heading_rad",5,0.,cfg.PSI_MIN,cfg.PSI_MAX,1e-7),
                   ("mass_kg",6,0.,cfg.MASS_MIN,m0,.01)]
    for name,j,offset,lo,hi,tolerance in state_specs:
        check(name,states[:,j]-offset,lambda t,j=j,offset=offset:float(dense(t)[j]-offset),
              lo,hi,tolerance)
    lon0 = np.radians(scenario.get("lon_0_deg",cfg.LON_0_DEG))
    lonf = np.radians(scenario.get("lon_f_deg",cfg.LON_F_DEG))
    lat0 = np.radians(scenario.get("lat_0_deg",cfg.LAT_0_DEG))
    latf = np.radians(scenario.get("lat_f_deg",cfg.LAT_F_DEG))
    check("longitude_rad",states[:,1],lambda t:float(dense(t)[1]),min(lon0,lonf)-.15,max(lon0,lonf)+.15,1e-9)
    check("latitude_rad",states[:,2],lambda t:float(dense(t)[2]),min(lat0,latf)-.15,max(lat0,latf)+.15,1e-9)
    for j,(name,lo,hi) in enumerate((("alpha_rad",cfg.ALPHA_MIN,cfg.ALPHA_MAX),
                                     ("bank_rad",cfg.SIGMA_MIN,cfg.SIGMA_MAX),
                                     ("throttle",cfg.ETA_MIN,cfg.ETA_MAX))):
        # Linear-control extrema occur at the knots exactly.
        low,high = float(sol["controls"][:,j].min()),float(sol["controls"][:,j].max())
        checks[name] = {"min":low,"max":high,"lower":float(lo),"upper":float(hi),
                        "tolerance":1e-9,"passed":bool(low>=lo-1e-9 and high<=hi+1e-9)}
    for name,key,lo,hi,tolerance in (("dynamic_pressure_pa","q",cfg.Q_MIN,cfg.Q_MAX,1.),
                                     ("normal_load","n_load",0.,cfg.N_MAX,1e-6),
                                     ("heat_flux_wm2","q_heat",0.,cfg.Q_HEAT_MAX,10.)):
        check(name,aux[key],lambda t,key=key:float(compute_dynamics(dense(t),control(t))[1][key]),
              lo,hi,tolerance)
    nfz_checks = {}
    for nfz in sol.get("nfzs",cfg.NO_FLY_ZONES):
        def clearance(t,nfz=nfz):
            x = dense(t)
            d = Constraints.no_fly_zone_distances_km(np.degrees(x[1]),np.degrees(x[2]),[nfz])
            return float(d[nfz["name"]]["clearance_km"]*1000.)
        values = Constraints.no_fly_zone_distances_km(np.degrees(states[:,1]),np.degrees(states[:,2]),[nfz])[nfz["name"]]["clearance_km"]*1000.
        minimum,_ = _extrema(time,values,clearance)
        nfz_checks[nfz["name"]] = {"min_clearance_m":minimum,"tolerance_m":.05,"passed":bool(minimum>=-.05)}

    node_sim = dense(sol["time"]).T
    error = node_sim-sol["states"]
    max_error = np.max(np.abs(error),axis=0)
    error_tolerance = np.array([.5,.5/cfg.R_E,.5/cfg.R_E,.02,np.radians(.002),np.radians(.002),.02])
    dynamics_passed = bool(np.isfinite(max_error).all() and np.all(max_error<=error_tolerance))
    # Compare physical equations through an independent ECEF force balance.
    reference_error = np.zeros(7)
    for x,u in zip(states[::max(1,len(time)//100)],controls[::max(1,len(time)//100)]):
        reference_error = np.maximum(reference_error,np.abs(compute_dynamics(x,u)[0]-compute_dynamics_reference(x,u)))
    reference_passed = bool(np.all(reference_error<1e-10))
    boundary_error = []
    for j,target in ((0,cfg.R_E+scenario.get("alt_0",cfg.ALT_0)),(1,lon0),(2,lat0),
                     (3,scenario.get("vel_0",cfg.VEL_0)),(4,scenario.get("gamma_0",cfg.GAMMA_0)),(6,m0)):
        boundary_error.append(abs(states[0,j]-target)/error_tolerance[j])
    for j,target in ((0,cfg.R_E+scenario.get("alt_f",cfg.ALT_F)),(1,lonf),(2,latf),
                     (3,scenario.get("vel_f",cfg.VEL_F)),(4,scenario.get("gamma_f",cfg.GAMMA_F))):
        boundary_error.append(abs(states[-1,j]-target)/error_tolerance[j])
    boundary_passed = bool(max(boundary_error)<=1.)
    time_passed = bool(cfg.TF_MIN-1e-6<=sol["tf"]<=cfg.TF_MAX+1e-6)
    solver_passed = bool(sol.get("solver_success",sol.get("success",False)))
    failures = [name for name,check_result in checks.items() if not check_result["passed"]]
    failures.extend("NFZ:"+name for name,check_result in nfz_checks.items() if not check_result["passed"])
    failures.extend(name for name,ok in (("dynamics_consistency",dynamics_passed),
                                        ("ECEF_reference",reference_passed),("mission_boundaries",boundary_passed),
                                        ("flight_time",time_passed),("solver",solver_passed)) if not ok)
    report = {"passed":not failures,"failures":failures,"checks":checks,"nfz_checks":nfz_checks,
              "min_clearance_m":min((c["min_clearance_m"] for c in nfz_checks.values()),default=None),
              "dynamics_max_abs_error_si":max_error.tolist(),"dynamics_tolerance_si":error_tolerance.tolist(),
              "terminal_error_si":error[-1].tolist(),"dynamics_passed":dynamics_passed,
              "reference_max_abs_error_si":reference_error.tolist(),"reference_passed":reference_passed,
              "boundary_passed":boundary_passed,"time_passed":time_passed,"solver_passed":solver_passed,
              "flight_time_tolerance_s":1e-6,
              "integration_method":method,"rtol":rtol,"atol":atol,"max_step_s":step,
              "grid_points":len(time),"extrema_refined":True}
    sol["_dense_solution"] = dense
    sol["_verification"] = {"time":np.asarray(sol["time"]),"states_sim":node_sim,
                             "states_opt":sol["states"],"errors":abs(error),"tf":sol["tf"],
                             "dense_time":time,"dense_states":states,"dense_controls":controls,
                             "validation":report,"passed":report["passed"]}
    sol["validation"] = report
    sol["success"] = bool(solver_passed and report["passed"])
    return report


def require_valid_solution(sol):
    if not sol.get("success",False) or not sol.get("validation",{}).get("passed",False):
        raise RuntimeError("Trajectory rejected: " + str(sol.get("return_status","unknown"))
                           + "; " + ", ".join(sol.get("validation",{}).get("failures",["not validated"])))


def save_solution(sol, path):
    require_valid_solution(sol)
    metadata = {k:sol[k] for k in ("success","solver_success","return_status","method",
                                   "shooting_intervals","rk4_substeps","model_version","solve_time")}
    np.savez_compressed(path,time=sol["time"],states=sol["states"],controls=sol["controls"],
                        scenario_json=json.dumps(sol["scenario"]),nfzs_json=json.dumps(sol["nfzs"]),
                        validation_json=json.dumps(sol["validation"]),metadata_json=json.dumps(metadata))


def load_solution(path):
    with np.load(path,allow_pickle=False) as data:
        sol = describe_trajectory(data["time"],data["states"],data["controls"],
                                  json.loads(str(data["nfzs_json"])),json.loads(str(data["scenario_json"])))
        metadata = json.loads(str(data["metadata_json"]))
        sol.update(metadata)
        # Acceptance is recomputed under the current model when a file is loaded.
    sol["validation"] = validate_solution(sol)
    sol["success"] = bool(sol["solver_success"] and sol["validation"]["passed"])
    return sol
