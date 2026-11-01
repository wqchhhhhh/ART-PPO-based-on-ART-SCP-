"""Generate continuously validated trajectories for sequence models."""

import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import sys
import time
import numpy as np

import config as cfg
from optimizer import TrajectoryOptimizer
from atmosphere import Atmosphere
from dataset_io import (HypersonicTrajectoryDataset,ChunkedHDF5Writer,build_sample,write_npz,SampleRejected)

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

MAX_NFZ=5
DEFAULT_SEQ_LEN=64


def great_circle_distance_deg(lon1, lat1, lon2, lat2):
    """Compute great circle distance [km] between two coordinates in degrees."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlam = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    return 2.0 * (cfg.R_E / 1000.0) * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _sample_endpoint(rng):
    """Joint altitude/speed sampling inside the terminal pressure/heat envelope."""
    while True:
        altitude=float(rng.uniform(28000.,32000.))
        rho=float(Atmosphere.density(altitude))
        heat_speed=(cfg.Q_HEAT_MAX/(cfg.K_HEAT*np.sqrt(rho/cfg.NOSE_RADIUS)))**(1./3.)
        lower=max(2050.,np.sqrt(2*cfg.Q_MIN/rho))
        upper=min(2180.,np.sqrt(2*cfg.Q_MAX/rho),heat_speed*.997)
        if upper>=lower:
            return altitude,float(rng.uniform(lower,upper))


def sample_random_mission(seed=None, max_nfz=MAX_NFZ):
    """
    Sample candidate missions with admissible terminal heat/pressure and NFZ clearance.
    Whole-trajectory feasibility is checked after optimization.
    """
    rng = np.random.default_rng(seed)

    # 1. Random Start Point
    lon_0_deg = float(rng.uniform(102.0, 108.0))
    lat_0_deg = float(rng.uniform(18.0, 24.0))
    alt_0,vel_0 = _sample_endpoint(rng)
    m_init = float(rng.uniform(19000.0, 21000.0))

    # 2. Random Target Point
    lon_f_deg = float(rng.uniform(138.0, 145.0))
    lat_f_deg = float(rng.uniform(28.0, 35.0))
    alt_f,vel_f = _sample_endpoint(rng)

    # 3. Random Number of No-Fly Zones (1 to 4)
    num_nfz = int(rng.integers(1, 5))
    num_nfz = min(num_nfz, max_nfz)

    # Direct line vector
    dlon = lon_f_deg - lon_0_deg
    dlat = lat_f_deg - lat_0_deg
    norm = np.hypot(dlon, dlat)
    n_lon = - dlat / (norm + 1e-12)
    n_lat =   dlon / (norm + 1e-12)

    # Stagger NFZ progress along flight corridor (between 20% and 80% of route)
    progress_fractions = np.sort(rng.uniform(0.20, 0.80, size=num_nfz))
    
    nfzs = []
    for k in range(num_nfz):
        s_k = progress_fractions[k]
        base_lon = lon_0_deg + s_k * dlon
        base_lat = lat_0_deg + s_k * dlat

        # Lateral perpendicular offset: [-160 km, +160 km]
        lat_offset_km = float(rng.uniform(-160.0, 160.0))
        offset_deg = lat_offset_km / 111.0

        nfz_lon = float(base_lon + offset_deg * n_lon)
        nfz_lat = float(base_lat + offset_deg * n_lat)
        radius_km = float(rng.uniform(140.0, 210.0))

        # Check distance to start and target
        d_start = great_circle_distance_deg(lon_0_deg, lat_0_deg, nfz_lon, nfz_lat)
        d_target = great_circle_distance_deg(lon_f_deg, lat_f_deg, nfz_lon, nfz_lat)

        # Enforce minimum distance from terminal points
        if d_start > radius_km + 100.0 and d_target > radius_km + 100.0:
            nfzs.append({
                "id": len(nfzs) + 1,
                "name": f"NFZ-{len(nfzs)+1}",
                "lon_deg": round(nfz_lon, 3),
                "lat_deg": round(nfz_lat, 3),
                "radius_km": round(radius_km, 1)
            })

    scenario = {
        "lon_0_deg": round(lon_0_deg, 3),
        "lat_0_deg": round(lat_0_deg, 3),
        "lon_f_deg": round(lon_f_deg, 3),
        "lat_f_deg": round(lat_f_deg, 3),
        "alt_0": round(alt_0, 1),
        "alt_f": round(alt_f, 1),
        "vel_0": round(vel_0, 1),
        "vel_f": round(vel_f, 1),
        "m_init": round(m_init, 1),
        "no_fly_zones": nfzs,
        "num_nfz": len(nfzs),
    }
    return scenario


def solve_mission_sample(sample_id,scenario,seq_len=DEFAULT_SEQ_LEN,max_nfz=MAX_NFZ,seed=-1):
    scenario=dict(scenario,shooting_intervals=seq_len-1,fixed_control_grid=True)
    optimizer=TrajectoryOptimizer(N_nodes=cfg.N_NODES,scenario_config=scenario)
    sol=optimizer.solve(max_iter=500,tol=1e-8,print_level=0)
    if not sol["success"]:
        print(f"Rejected sample {sample_id} (seed={seed}): {sol['return_status']}; "
              f"{sol.get('validation',{}).get('failures',[])}",flush=True)
        return None
    try:
        return build_sample(sol,sample_id,seq_len,max_nfz,seed)
    except SampleRejected as error:
        print(f"Rejected sample {sample_id} (seed={seed}): {error}",flush=True)
        return None


def worker_task(args):
    sample_id,seed,seq_len,max_nfz=args
    scenario=sample_random_mission(seed=seed,max_nfz=max_nfz)
    return solve_mission_sample(sample_id,scenario,seq_len,max_nfz,seed)


def iter_validated_samples(count,num_workers,seq_len,max_nfz,start_seed=100,stride=13,
                           start_id=0,max_attempts=None):
    if count<1 or num_workers<1 or seq_len<2:
        raise ValueError("count/workers must be positive and seq_len must be >=2")
    limit=max_attempts if max_attempts is not None else max(20,count*20)
    attempts=accepted=0
    executor=ProcessPoolExecutor(max_workers=num_workers) if num_workers>1 else None
    try:
        if executor is None:
            while accepted<count and attempts<limit:
                args=(start_id+attempts,start_seed+attempts*stride,seq_len,max_nfz)
                attempts+=1
                sample=worker_task(args)
                if sample is not None:
                    accepted+=1
                    yield sample
        else:
            pending=set()
            while accepted<count:
                # Keep workers occupied as soon as a candidate finishes. The
                # remaining target bounds in-flight work, so no extra valid
                # trajectories are generated at the end of the run.
                while len(pending)<min(num_workers,count-accepted) and attempts<limit:
                    args=(start_id+attempts,start_seed+attempts*stride,seq_len,max_nfz)
                    pending.add(executor.submit(worker_task,args))
                    attempts+=1
                if not pending:
                    break
                finished,_=wait(pending,return_when=FIRST_COMPLETED)
                for future in finished:
                    pending.remove(future)
                    sample=future.result()
                    if sample is not None:
                        accepted+=1
                        yield sample
        if accepted<count:
            raise RuntimeError(f"Collected {accepted}/{count} validated samples in {attempts} attempts")
    finally:
        if executor is not None:
            executor.shutdown(wait=True,cancel_futures=True)


def generate_dataset(num_samples=100,num_workers=4,seq_len=DEFAULT_SEQ_LEN,output_dir="dataset",
                     filename_prefix="validated_trajectory",max_attempts=None):
    destination=Path(output_dir)
    if not destination.is_dir():
        raise ValueError(f"Create/select the output directory before generation: {destination}")
    h5_path=destination/f"{filename_prefix}_{num_samples}.h5"
    npz_path=destination/f"{filename_prefix}_{num_samples}.npz"
    if h5_path.exists() or npz_path.exists():
        raise FileExistsError("Choose a new filename_prefix for a fresh dataset")
    writer=ChunkedHDF5Writer(h5_path,seq_len,MAX_NFZ)
    samples=[]
    started=time.perf_counter()
    try:
        for sample in iter_validated_samples(num_samples,num_workers,seq_len,MAX_NFZ,max_attempts=max_attempts):
            writer.add_sample(sample)
            samples.append(sample)
            print(f"Validated samples: {len(samples)}/{num_samples}",flush=True)
    finally:
        writer.flush()
    write_npz(npz_path,samples,seq_len,MAX_NFZ)
    print(f"Saved {len(samples)} validated samples in {time.perf_counter()-started:.1f} s")
    return str(npz_path),str(h5_path)


def inspect_dataset(filepath,plot_samples=True,allow_unvalidated=False):
    dataset=HypersonicTrajectoryDataset(filepath,allow_unvalidated=allow_unvalidated)
    print(f"Samples={len(dataset)}, validated={dataset.validated}, controls={dataset.controls.shape}")
    print(f"Fuel range={dataset.fuel_burneds.min():.3f}..{dataset.fuel_burneds.max():.3f} kg")
    if plot_samples:
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(10,6))
        for j in range(min(8,len(dataset))):
            sample=dataset[j]
            ax.plot(sample["states"][:,1],sample["states"][:,2],label=f"seed={sample.get('seed',j)}")
        ax.set(xlabel="Longitude [deg]",ylabel="Latitude [deg]",title="Validated generated trajectories")
        ax.grid(alpha=.3)
        ax.legend()
        fig.savefig(str(Path(filepath).with_suffix(".png")),dpi=160,bbox_inches="tight")
        plt.close(fig)
    return dataset


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_samples",type=int,default=20)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seq_len",type=int,default=64)
    parser.add_argument("--output_dir",default="dataset")
    parser.add_argument("--prefix",default="validated_trajectory")
    parser.add_argument("--max_attempts",type=int)
    args=parser.parse_args()
    npz,h5=generate_dataset(args.num_samples,args.workers,args.seq_len,args.output_dir,args.prefix,args.max_attempts)
    inspect_dataset(npz)
