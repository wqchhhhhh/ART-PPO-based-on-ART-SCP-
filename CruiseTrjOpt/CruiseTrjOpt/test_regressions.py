"""Regression checks for physical equations, continuous acceptance and storage.

Run after scenario_experiment.py using python -m unittest -v test_regressions.
"""

import contextlib
import copy
import hashlib
import io
from pathlib import Path
import unittest
import uuid
import h5py
import casadi as ca
import numpy as np

import config as cfg
from atmosphere import Atmosphere
from collocation import lgl_nodes_weights_diff,interpolate_trajectory
from constraints import Constraints
from continuous import load_solution,validate_solution,require_valid_solution
from dataset_io import (build_sample,decode_physical,ChunkedHDF5Writer,
                        HypersonicTrajectoryDataset,write_npz)
from dynamics import compute_dynamics,compute_dynamics_reference,get_casadi_dynamics_function
from optimizer import TrajectoryOptimizer
from verify_dynamics import forward_integrate_trajectory,print_verification_report
from main import print_summary_report,save_trajectory_csv
from generate_dataset import sample_random_mission


def standard_atmosphere(h):
    """US76 reference layers, evaluated without smoothing."""
    height=6356766.*h/(6356766.+h)
    boundaries=[0.,11000.,20000.,32000.,47000.]
    lapse=[-.0065,0.,.001,.0028]
    temperature,pressure=288.15,101325.
    for j,L in enumerate(lapse):
        dh=min(height,boundaries[j+1])-boundaries[j]
        if L==0:
            pressure*=np.exp(-9.80665*dh/(287.05287*temperature))
        else:
            pressure*=(temperature/(temperature+L*dh))**(9.80665/(287.05287*L))
        temperature+=L*dh
        if height<=boundaries[j+1]:
            break
    return temperature,pressure/(287.05287*temperature)


class PhysicsTests(unittest.TestCase):
    def test_rotating_spherical_equations_against_ecef(self):
        rng=np.random.default_rng(12)
        for _ in range(200):
            x=np.array([cfg.R_E+rng.uniform(25000,35000),rng.uniform(-2.5,2.5),
                         rng.uniform(-1.,1.),rng.uniform(1850,2350),rng.uniform(-.0698,.0698),
                         rng.uniform(-np.pi,np.pi),rng.uniform(12000,21000)])
            u=np.array([rng.uniform(-.026,.139),rng.uniform(-1.04,1.04),rng.uniform(.05,1.)])
            np.testing.assert_allclose(compute_dynamics(x,u)[0],compute_dynamics_reference(x,u),rtol=1e-12,atol=1e-12)

    def test_symbolic_numeric_and_dm_agree(self):
        x=np.array([cfg.R_E+30000,2.,.5,2100.,.06,1.,18000.])
        u=np.array([.08,.2,.7])
        expected=compute_dynamics(x,u)[0]
        np.testing.assert_allclose(np.array(get_casadi_dynamics_function()(x,u)[0]).ravel(),expected,atol=1e-12)
        np.testing.assert_allclose(np.array(compute_dynamics(ca.DM(x),ca.DM(u))[0]).ravel(),expected,atol=1e-12)

    def test_us76_density_fit_accuracy(self):
        height=np.linspace(20000.,40000.,2001)
        reference=np.array([standard_atmosphere(h)[1] for h in height])
        relative=np.max(abs(Atmosphere.density(height)/reference-1))
        self.assertLess(relative,2e-4)

    def test_us76_geometric_height_and_temperature(self):
        for h in (25000.,30000.,35000.):
            temperature,density=standard_atmosphere(h)
            self.assertAlmostEqual(float(Atmosphere.temperature(h)),temperature,places=5)
            self.assertLess(abs(float(Atmosphere.density(h))/density-1),3e-6)

    def test_lgl_polynomial_differentiation_and_quadrature(self):
        nodes,weights,D=lgl_nodes_weights_diff(12)
        for power in range(13):
            expected=power*nodes**(power-1) if power else np.zeros_like(nodes)
            np.testing.assert_allclose(D@(nodes**power),expected,atol=1e-10)
        for power in range(24):
            integral=0. if power%2 else 2./(power+1)
            self.assertAlmostEqual(float(weights@(nodes**power)),integral,places=12)

    def test_interpolated_controls_preserve_bounds(self):
        t=np.array([0.,.01,1.,2.,9.])
        u=np.array([[1.,-.5,.05],[.4,.5,1.],[.2,-.2,.05],[.3,.2,1.],[.1,0.,.05]])
        sampled=interpolate_trajectory(t,u,np.linspace(0,9,10001))
        self.assertTrue(np.all(sampled>=u.min(axis=0)-1e-14))
        self.assertTrue(np.all(sampled<=u.max(axis=0)+1e-14))

    def test_scaled_nfz_has_physical_distance_scale(self):
        zone=Constraints.prepare_nfzs([dict(name="test",lon_deg=0.,lat_deg=0.,radius_km=180.)])
        residual=float(Constraints.scaled_no_fly_zone_residuals(179980./cfg.R_E,0.,zone))
        self.assertAlmostEqual(residual,.020,places=5)

    def test_configured_time_upper_bound(self):
        optimizer=TrajectoryOptimizer(N_nodes=5)
        self.assertEqual(float(optimizer.build_nlp()[2][-1]*optimizer.tf_scale),cfg.TF_MAX)

    def test_random_mission_terminal_envelopes(self):
        for seed in range(100):
            mission=sample_random_mission(seed)
            for endpoint in ("0","f"):
                h,V=mission["alt_"+endpoint],mission["vel_"+endpoint]
                q=Atmosphere.dynamic_pressure(h,V)
                heat=cfg.K_HEAT*np.sqrt(Atmosphere.density(h)/cfg.NOSE_RADIUS)*V**3
                self.assertGreaterEqual(q,cfg.Q_MIN)
                self.assertLessEqual(q,cfg.Q_MAX)
                self.assertLessEqual(heat,cfg.Q_HEAT_MAX)


class ContinuousAndStorageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sol=load_solution("results/scenario_1_solution.npz")
        require_valid_solution(cls.sol)
        cls.T=len(cls.sol["time"])
        cls.sample=build_sample(cls.sol,7,seq_len=cls.T,max_nfz=5,seed=104)

    def setUp(self):
        self.created=[]

    def tearDown(self):
        for path in self.created:
            path.unlink(missing_ok=True)

    def filename(self,suffix):
        path=Path("results")/("_regression_"+uuid.uuid4().hex+suffix)
        self.created.append(path)
        return path

    def test_baseline_continuous_checks(self):
        self.assertTrue(self.sol["validation"]["passed"])
        self.assertGreater(self.sol["validation"]["min_clearance_m"],0.)
        self.assertTrue(all(item["passed"] for item in self.sol["validation"]["checks"].values()))

    def test_large_state_error_fails_verification(self):
        bad=dict(self.sol,states=self.sol["states"].copy())
        bad["states"][1:,0]+=10000.
        data=forward_integrate_trajectory(bad)
        self.assertFalse(data["passed"])
        self.assertFalse(bad["success"])
        with self.assertRaises(RuntimeError):
            require_valid_solution(bad)
        self.assertIn("dynamics_consistency",data["validation"]["failures"])
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertFalse(print_verification_report(data))
        self.assertIn("verification: FAIL",output.getvalue())

    def test_failed_solver_cannot_publish(self):
        bad=dict(self.sol,success=False,solver_success=False,return_status="Maximum_Iterations_Exceeded")
        bad.pop("_verification",None)
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertFalse(print_summary_report(bad))
        self.assertNotIn("[PASS]",output.getvalue())
        with self.assertRaises(RuntimeError):
            save_trajectory_csv(bad,self.filename(".csv"))

    def test_stored_derivatives_match_stored_controls(self):
        x,u=decode_physical(self.sample["states"],self.sample["controls"])
        derivative,aux=compute_dynamics(x.T,u.T)
        np.testing.assert_allclose(self.sample["derivatives"],derivative.T,rtol=1e-6,atol=1e-7)
        np.testing.assert_allclose(self.sample["telemetry"][:,0],aux["q"],rtol=1e-6)

    def test_serialized_control_law_passes_independent_replay(self):
        x,u=decode_physical(self.sample["states"],self.sample["controls"])
        time=self.sample["time_norm"].astype(float)*self.sample["tf"]
        stored_zones=[dict(name=self.sol["nfzs"][i]["name"],lon_deg=float(zone[0]),
                           lat_deg=float(zone[1]),radius_km=float(zone[2]))
                      for i,zone in enumerate(self.sample["nfz"]) if self.sample["nfz_mask"][i]]
        exported=dict(self.sol,time=time,states=x,controls=u,nfzs=Constraints.prepare_nfzs(stored_zones))
        self.assertTrue(validate_solution(exported)["passed"])
        np.testing.assert_allclose(exported["validation"]["min_clearance_m"],self.sample["min_clearance_m"],atol=1e-5)

    def test_hdf5_and_npz_use_identical_schema(self):
        h5,npz=self.filename(".h5"),self.filename(".npz")
        writer=ChunkedHDF5Writer(h5,self.T,5,chunk_size=1)
        writer.add_sample(self.sample)
        write_npz(npz,[self.sample],self.T,5)
        a,b=HypersonicTrajectoryDataset(h5),HypersonicTrajectoryDataset(npz)
        self.assertTrue(a.validated and b.validated)
        self.assertEqual(set(a[0]),set(b[0]))
        for key in a[0]:
            np.testing.assert_array_equal(a[0][key],b[0][key])
        with h5py.File(h5,"r") as f:
            self.assertEqual(f["states"].dtype,np.dtype("float64"))
            self.assertEqual(f["states"].compression,"gzip")

    def test_resume_mismatch_preserves_existing_bytes(self):
        path=self.filename(".h5")
        writer=ChunkedHDF5Writer(path,self.T,5,chunk_size=1)
        writer.add_sample(self.sample)
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            ChunkedHDF5Writer(path,self.T-1,5)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)
        with h5py.File(path,"r") as f:
            self.assertEqual(f["states"].shape,(1,self.T,7))

    def test_incomplete_append_recovers_committed_rows(self):
        path=self.filename(".h5")
        writer=ChunkedHDF5Writer(path,self.T,5,chunk_size=1)
        writer.add_sample(self.sample)
        with h5py.File(path,"a") as f:
            f["states"].resize(3,axis=0)
        resumed=ChunkedHDF5Writer(path,self.T,5)
        self.assertEqual(resumed.get_existing_count(),1)
        with h5py.File(path,"r") as f:
            self.assertEqual(f["states"].shape,(1,self.T,7))
        self.assertEqual(resumed.next_seed(start=100,stride=11),115)

    def test_rejected_sample_and_wrong_shape_are_blocked(self):
        path=self.filename(".h5")
        writer=ChunkedHDF5Writer(path,self.T,5)
        bad=dict(self.sample,valid=False)
        with self.assertRaises(ValueError):
            writer.add_sample(bad)
        bad=dict(self.sample,states=np.zeros((self.T-1,7)))
        with self.assertRaises(ValueError):
            writer.add_sample(bad)
        self.assertEqual(writer.get_existing_count(),0)

    def test_resumed_append_preserves_prior_sample(self):
        path=self.filename(".h5")
        writer=ChunkedHDF5Writer(path,self.T,5,chunk_size=1)
        writer.add_sample(self.sample)
        resumed=ChunkedHDF5Writer(path,self.T,5,chunk_size=1)
        resumed.add_sample(dict(self.sample,seed=np.int64(115),sample_id=np.int64(8)))
        self.assertEqual(resumed.get_existing_count(),2)
        loaded=HypersonicTrajectoryDataset(path)
        np.testing.assert_array_equal(loaded[0]["states"],self.sample["states"])
        self.assertEqual(int(loaded[1]["seed"]),115)

    def test_legacy_files_require_explicit_inspection(self):
        path=self.filename(".npz")
        np.savez(path,condition_states=self.sample["condition"][None],nfz_arrays=self.sample["nfz"][None],
                 nfz_masks=self.sample["nfz_mask"][None],controls=self.sample["controls"][None],
                 states=self.sample["states"][None],tfs=np.array([self.sample["tf"]]),
                 fuel_burneds=np.array([self.sample["fuel_burned"]]))
        with self.assertRaises(ValueError):
            HypersonicTrajectoryDataset(path)
        legacy=HypersonicTrajectoryDataset(path,allow_unvalidated=True)
        self.assertFalse(legacy.validated)
        self.assertFalse(legacy[0]["valid"])


if __name__=="__main__":
    unittest.main(verbosity=2)
