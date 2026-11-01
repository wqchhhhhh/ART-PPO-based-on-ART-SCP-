"""Validated trajectory tensors, versioned storage and a shared loader."""

import json
from pathlib import Path
import h5py
import numpy as np

import config as cfg
from continuous import sample_solution,require_valid_solution,validate_solution
from dynamics import compute_dynamics
from constraints import Constraints

SCHEMA_VERSION = 2


class SampleRejected(ValueError):
    """A candidate failed export-time continuous verification."""


UNITS = {"condition":"[km,deg,deg,100 m/s,10000 kg,km,deg,deg,100 m/s]",
         "nfz":"[deg,deg,km]","controls":"[deg,deg,1]",
         "states":"[km,deg,deg,m/s,deg,deg,kg]",
         "telemetry":"[Pa,1,W/m2,1,N,N]",
         "derivatives":"[m/s,rad/s,rad/s,m/s2,rad/s,rad/s,kg/s]",
         "time_norm":"1","tf":"s","fuel_burned":"kg"}


def tensor_shapes(seq_len,max_nfz):
    return {"condition":(9,),"nfz":(max_nfz,3),"nfz_mask":(max_nfz,),
            "controls":(seq_len,3),"states":(seq_len,7),"telemetry":(seq_len,6),
            "derivatives":(seq_len,7),"time_norm":(seq_len,),"tf":(),"fuel_burned":(),
            "seed":(),"sample_id":(),"valid":(),"min_clearance_m":(),"dynamics_max_abs_error":(7,)}


def metadata(seq_len,max_nfz):
    return {"schema_version":SCHEMA_VERSION,"model_version":cfg.MODEL_VERSION,
            "seq_len":seq_len,"max_nfz":max_nfz,"continuous_validation":True,
            "control_interpolation":"piecewise_linear","state_source":"DOP853 forward integration",
            "validation_max_step_s":cfg.VALIDATION_MAX_STEP,"units_json":json.dumps(UNITS)}


def decode_physical(states,controls):
    x,u = np.asarray(states,dtype=float).copy(),np.asarray(controls,dtype=float).copy()
    x[...,0] = cfg.R_E+x[...,0]*1000.
    x[...,[1,2,4,5]] = np.radians(x[...,[1,2,4,5]])
    u[...,:2] = np.radians(u[...,:2])
    return x,u


def build_sample(sol,sample_id,seq_len=128,max_nfz=5,seed=-1):
    require_valid_solution(sol)
    if seq_len<2 or len(sol["nfzs"])>max_nfz:
        raise ValueError("Sequence length must be >=2 and max_nfz must cover the mission")
    tau = np.linspace(0.,1.,seq_len).astype(np.float32)
    time = tau.astype(float)*sol["tf"]
    x,u = sample_solution(sol,time)
    states = x.copy()
    states[:,0] = (states[:,0]-cfg.R_E)/1000.
    states[:,[1,2,4,5]] = np.degrees(states[:,[1,2,4,5]])
    controls = u.copy()
    controls[:,:2] = np.degrees(controls[:,:2])
    states,controls = states.astype(np.float64),controls.astype(np.float32)
    # A uniform control sequence defines a new control law unless its knots
    # coincide with the optimizer's grid. Revalidate the stored precision and
    # grid, then use its propagated states for all exported physics channels.
    _,stored_u = decode_physical(states,controls)
    nfz = np.zeros((max_nfz,3),np.float32)
    mask = np.zeros(max_nfz,bool)
    stored_zones = []
    for k,zone in enumerate(sol["nfzs"]):
        nfz[k] = [zone["lon_deg"],zone["lat_deg"],zone["radius_km"]]
        mask[k] = True
        stored_zones.append(dict(name=zone["name"],lon_deg=float(nfz[k,0]),
                                 lat_deg=float(nfz[k,1]),radius_km=float(nfz[k,2])))
    exported = dict(sol,time=time,states=x.copy(),controls=stored_u,
                    nfzs=Constraints.prepare_nfzs(stored_zones))
    validation = validate_solution(exported)
    if not validation["passed"]:
        raise SampleRejected("Stored control sequence failed continuous verification: "+", ".join(validation["failures"]))
    x = exported["_dense_solution"](time).T
    states = x.copy()
    states[:,0] = (states[:,0]-cfg.R_E)/1000.
    states[:,[1,2,4,5]] = np.degrees(states[:,[1,2,4,5]])
    # Derivatives and telemetry are computed from the exact stored values.
    # Quantization is the only representation change; controls are never clipped.
    physical_x,physical_u = decode_physical(states,controls)
    dx,aux = compute_dynamics(physical_x.T,physical_u.T)
    telemetry = np.column_stack([aux[k] for k in ("q","n_load","q_heat","Ma","L","D")]).astype(np.float32)
    condition = np.array([(x[0,0]-cfg.R_E)/1000.,np.degrees(x[0,1]),np.degrees(x[0,2]),
                           x[0,3]/100.,x[0,6]/10000.,(x[-1,0]-cfg.R_E)/1000.,
                           np.degrees(x[-1,1]),np.degrees(x[-1,2]),x[-1,3]/100.],np.float64)
    return {"condition":condition,"nfz":nfz,"nfz_mask":mask,"states":states,"controls":controls,
            "telemetry":telemetry,"derivatives":dx.T.astype(np.float32),"time_norm":tau,
            "tf":np.float64(sol["tf"]),"fuel_burned":np.float64(x[0,6]-x[-1,6]),
            "seed":np.int64(seed),"sample_id":np.int64(sample_id),"valid":np.bool_(True),
            "min_clearance_m":np.float64(validation["min_clearance_m"] if validation["min_clearance_m"] is not None else np.nan),
            "dynamics_max_abs_error":np.array(validation["dynamics_max_abs_error_si"],float)}


def write_npz(path,samples,seq_len,max_nfz):
    arrays = {k:np.stack([s[k] for s in samples]) for k in tensor_shapes(seq_len,max_nfz)}
    np.savez_compressed(path,**arrays,metadata_json=json.dumps(metadata(seq_len,max_nfz)))


class ChunkedHDF5Writer:
    """Append rows, validate the schema before writing, commit complete batches."""
    def __init__(self,filepath,seq_len=128,max_nfz=5,chunk_size=50):
        if seq_len<2 or max_nfz<0 or chunk_size<1:
            raise ValueError("seq_len must be >=2, max_nfz >=0 and chunk_size >=1")
        self.filepath = str(filepath)
        self.seq_len,self.max_nfz,self.chunk_size = seq_len,max_nfz,chunk_size
        self.shapes = tensor_shapes(seq_len,max_nfz)
        self.buffer = []
        if Path(filepath).exists():
            # Check all non-sample dimensions and model metadata before mutation.
            with h5py.File(filepath,"r") as hf:
                for k in ("schema_version","model_version","seq_len","max_nfz","continuous_validation"):
                    if hf.attrs.get(k)!=metadata(seq_len,max_nfz)[k]:
                        raise ValueError(f"Cannot resume: incompatible {k} in {filepath}")
                committed = int(hf.attrs["committed_count"])
                for key,shape in self.shapes.items():
                    if key not in hf or hf[key].shape[1:]!=shape or len(hf[key])<committed:
                        raise ValueError(f"Cannot resume: incompatible field {key}")
            with h5py.File(filepath,"a") as hf:
                for key in self.shapes:
                    # Recover only an uncommitted tail after an interrupted append.
                    hf[key].resize(committed,axis=0)
        else:
            with h5py.File(filepath,"x") as hf:
                for k,v in metadata(seq_len,max_nfz).items():
                    hf.attrs[k] = v
                hf.attrs["committed_count"] = 0
                for key,shape in self.shapes.items():
                    dtype = (np.bool_ if key in ("nfz_mask","valid") else np.int64
                             if key in ("seed","sample_id") else np.float64
                             if key in ("condition","states","tf","fuel_burned","min_clearance_m","dynamics_max_abs_error") else np.float32)
                    hf.create_dataset(key,shape=(0,*shape),maxshape=(None,*shape),dtype=dtype,
                                      chunks=(chunk_size,*shape),compression="gzip",compression_opts=4,shuffle=True)

    def get_existing_count(self):
        with h5py.File(self.filepath,"r") as hf:
            return int(hf.attrs["committed_count"])

    def next_seed(self,start=5000,stride=11):
        with h5py.File(self.filepath,"r") as hf:
            count = int(hf.attrs["committed_count"])
            return max(start,int(hf["seed"][:count].max())+stride) if count else start

    def add_sample(self,sample):
        if not sample["valid"]:
            raise ValueError("Only validated samples may be stored")
        for key,shape in self.shapes.items():
            if np.shape(sample[key])!=shape:
                raise ValueError(f"Incorrect sample shape for {key}")
        self.buffer.append(sample)
        if len(self.buffer)>=self.chunk_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        batches = {k:np.stack([s[k] for s in self.buffer]) for k in self.shapes}
        with h5py.File(self.filepath,"a") as hf:
            start = int(hf.attrs["committed_count"])
            stop = start+len(self.buffer)
            for key,batch in batches.items():
                hf[key].resize(stop,axis=0)
                hf[key][start:stop] = batch
            hf.flush()
            hf.attrs["committed_count"] = stop
            hf.flush()
        self.buffer.clear()


class HypersonicTrajectoryDataset:
    """PyTorch-compatible tensors; both historic field spellings can be inspected.

    Legacy files require allow_unvalidated=True and expose valid=False. Files
    written by the current generators use one shared schema with validation metadata.
    """
    ALIASES = {"condition":"condition_states","nfz":"nfz_arrays","nfz_mask":"nfz_masks",
               "tf":"tfs","fuel_burned":"fuel_burneds","time_norm":"time_norms"}

    def __init__(self,filepath,allow_unvalidated=False):
        filepath = str(filepath)
        if filepath.endswith(".npz"):
            with np.load(filepath,allow_pickle=False) as file:
                self.metadata = json.loads(str(file["metadata_json"])) if "metadata_json" in file else {}
                arrays = {k:file[k] for k in file.files if k!="metadata_json"}
        elif filepath.endswith((".h5",".hdf5")):
            with h5py.File(filepath,"r") as file:
                self.metadata = dict(file.attrs)
                count = int(file.attrs.get("committed_count",len(file["controls"])))
                arrays = {k:v[:count] for k,v in file.items()}
        else:
            raise ValueError("Supported formats: .npz, .h5, .hdf5")
        self.validated = bool(self.metadata.get("schema_version")==SCHEMA_VERSION
                              and self.metadata.get("model_version")==cfg.MODEL_VERSION
                              and self.metadata.get("continuous_validation",False))
        if not self.validated and not allow_unvalidated:
            raise ValueError("Dataset has no current continuous validation metadata; use allow_unvalidated=True for inspection")
        self.arrays = {}
        for key in ("condition","nfz","nfz_mask","controls","states","tf","fuel_burned"):
            source = key if key in arrays else self.ALIASES.get(key,key)
            self.arrays[key] = arrays[source]
        for key in ("telemetry","derivatives","time_norm","seed","sample_id","min_clearance_m","dynamics_max_abs_error"):
            source = key if key in arrays else self.ALIASES.get(key,key)
            if source in arrays:
                self.arrays[key] = arrays[source]
        self.arrays["valid"] = arrays.get("valid",np.zeros(len(self.arrays["controls"]),bool))
        if self.validated and not self.arrays["valid"].all():
            raise ValueError("Dataset contains rejected samples")
        self.condition_states,self.nfz_arrays,self.nfz_masks = (self.arrays[k] for k in ("condition","nfz","nfz_mask"))
        self.controls,self.states,self.tfs,self.fuel_burneds = (self.arrays[k] for k in ("controls","states","tf","fuel_burned"))

    def __len__(self):
        return len(self.controls)

    def __getitem__(self,index):
        return {k:v[index] for k,v in self.arrays.items()}
