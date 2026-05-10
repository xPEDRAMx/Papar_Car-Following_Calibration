"""
Prospect Theory (PT) Calibration on TGSIM using the paper's methodology:

- Extract valid car-following episodes:
  * duration > 10 s
  * headway < 200 m
  * no lane change during the episode
  * preceding vehicle ID remains unchanged (leader constant in the episode)
- Calibrate PT parameters for EACH leader–follower episode via Genetic Algorithm (GA)
- Fitness = sum_j [ w_pos*|x_obs - x_sim| + w_speed*|v_obs - v_sim| ]

Paper basis: Prospect Theory car-following model
"""

from __future__ import annotations

import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd

# Try to import numba for JIT compilation (much faster)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create a dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# Stats tests
try:
    from scipy import stats
    SCIPY_STATS_AVAILABLE = True
except ImportError:
    SCIPY_STATS_AVAILABLE = False

# -----------------------------
# Logging setup - Tee class to write to both console and file
# -----------------------------
class Tee:
    """Write to both console and file simultaneously"""
    def __init__(self, file_path: str):
        self.file = open(file_path, 'w', encoding='utf-8')
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, text: str):
        self.stdout.write(text)
        self.file.write(text)
        self.file.flush()  # Ensure immediate write

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()

# -----------------------------
# CONFIG (edit these)
# -----------------------------
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "Dataset")

# Define all dataset paths
CSV_PATHS = [
    os.path.join(DATASET_DIR, r"Third_Generation_Simulation_Data__TGSIM__I-395_Trajectories.csv"),
    # Disabled for now (I-90/I-94 geometry is diagonal; longitudinal axis needs projection)
    # os.path.join(DATASET_DIR, r"Third_Generation_Simulation_Data__TGSIM__I-90_I-94_Stationary_Trajectories.csv"),
    os.path.join(DATASET_DIR, r"Third_Generation_Simulation_Data__TGSIM__I-294_L1_Trajectories.csv"),
    os.path.join(DATASET_DIR, r"Third_Generation_Simulation_Data__TGSIM__I-294_L2_Trajectories.csv"),
]

# Dataset names for tracking
DATASET_NAMES = [
    "I-395",
    # Disabled for now: "I-90_I-94",
    "I-294_L1",
    "I-294_L2",
]

# Calibration settings
CALIBRATE_ONLY_NEAR_AVS = True  # If True: only calibrate vehicles near AVs (equal sampling)
                                 # If False: calibrate ALL episodes (no filtering)

# Results directory: "Results PT" for equal sampling, "Results Total PT" for all episodes
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results PT" if CALIBRATE_ONLY_NEAR_AVS else "Results Total PT")

OUTPUT_EPISODES_CSV = os.path.join(RESULTS_DIR, "pt_calib_episodes_results.csv")
OUTPUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "pt_calib_vehicle_type_summary.csv")
OUTPUT_EPISODES_EXCEL = os.path.join(RESULTS_DIR, "pt_calib_episodes_summary.xlsx")
PLOT_COMPARISONS = True  # Whether to create comparison plots

# Vehicle type mapping
# Type 1: small cars
# Type 2: trucks -> large
# Type 3: buses -> large
# Type 4: autonomous vehicles (AV)
VEHICLE_TYPE_MAP = {
    1: "small",
    2: "large",  # trucks
    3: "large",  # buses
    4: "av",     # autonomous vehicles
}

# Fitness weights
W_POS = 1.0
W_SPEED = 0

# Episode constraints from paper
MIN_EPISODE_DURATION_S = 10
MAX_HEADWAY_M = 200.0

# Integration settings
DT_MIN = 0.02
DT_MAX = 0.30

# GA settings (optimized for speed; increase if you want tighter calibration)
# Reduced for PT model (more expensive per evaluation)
# For even faster calibration, try: GA_POP=20, GA_GENS=30
GA_POP = 30  # Reduced from 50 (original IDM had 50)
GA_GENS = 50  # Reduced from 80 (original IDM had 80)
GA_ELITE_FRAC = 0.15
GA_TOURN_K = 3
GA_CROSSOVER_PROB = 0.9
GA_MUTATION_PROB = 0.25
GA_MUTATION_SCALE = 0.15  # fraction of parameter range for gaussian mutation
GA_EARLY_STOP_GENS = 8  # Reduced from 10 for faster stopping
GA_EARLY_STOP_TOL = 1e-5  # Slightly relaxed from 1e-6

# Random seed for reproducibility (set to None for non-deterministic results)
RANDOM_SEED = 42  # Change this value to get different but reproducible results

# Multiple runs configuration for robust calibration
# If > 1: run calibration N times with different seeds and aggregate results
# If 1: single run (faster, but less robust)
N_CALIBRATION_RUNS = 20  # Recommended: 10-30 for robust results, 1 for speed
USE_BEST_RUN = True  # If True: use best run (lowest fitness). If False: use mean of all runs

# Parallel processing configuration
# Number of parallel workers for episode calibration (None = use all CPU cores, 1 = no parallelization)
# IMPORTANT: keep this as None or an int (not a string).
N_PARALLEL_WORKERS = 12  # Set to None for auto (uses all cores), or specify number (e.g., 4)
# Note: Parallelization gives ~4-8x speedup on multi-core CPUs, but uses more memory
# 
# GPU acceleration note: GPU would require rewriting the PT model in JAX/CuPy, but the Newton-Raphson
# solver is iterative/sequential, so GPU benefits would be limited. Multiprocessing is more practical
# since each episode calibration is independent and can run in parallel on CPU cores.

# Override from environment (used by run_calibration_sweep.py for W_POS/W_SPEED combos)
if "CALIB_W_POS" in os.environ:
    W_POS = float(os.environ["CALIB_W_POS"])
if "CALIB_W_SPEED" in os.environ:
    W_SPEED = float(os.environ["CALIB_W_SPEED"])
if "CALIB_OUTPUT_SUBFOLDER" in os.environ:
    RESULTS_DIR = os.path.join(SCRIPT_DIR, os.environ["CALIB_OUTPUT_SUBFOLDER"])
    OUTPUT_EPISODES_CSV = os.path.join(RESULTS_DIR, "pt_calib_episodes_results.csv")
    OUTPUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "pt_calib_vehicle_type_summary.csv")
    OUTPUT_EPISODES_EXCEL = os.path.join(RESULTS_DIR, "pt_calib_episodes_summary.xlsx")

# -----------------------------
# Column guessing helpers
# -----------------------------
def guess_column(cols: List[str], candidates: List[str]) -> Optional[str]:
    cols_lower = {c.lower(): c for c in cols}

    # exact matches
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]

    # boundary-ish partial matches
    for cand in candidates:
        cand_lower = cand.lower()
        for c in cols:
            c_lower = c.lower()
            if c_lower == cand_lower or c_lower.startswith(cand_lower + '_') or c_lower.startswith(cand_lower + '-'):
                return c

    # fallback substring
    for cand in candidates:
        cand_lower = cand.lower()
        for c in cols:
            if cand_lower in c.lower():
                return c
    return None


@dataclass
class Schema:
    time: str
    veh_id: str
    lead_id: Optional[str]  # optional; will be computed if missing
    lane: str
    speed: str
    pos: str
    veh_type: str
    run_index: Optional[str] = None  # optional; separates distinct data-collection runs
    av_column: Optional[str] = None  # optional; separate AV column
    length: Optional[str] = None     # optional
    # True = position increases in direction of travel (leader has larger pos); False = reversed axis
    pos_increases_downstream: bool = True


def infer_schema(df: pd.DataFrame, dataset_name: str = None) -> Schema:
    cols = list(df.columns)
    time = guess_column(cols, ["time", "t", "timestamp", "sec", "seconds"])
    veh_id = guess_column(cols, ["veh_id", "vehicle_id", "id", "track_id"])
    lead_id = guess_column(cols, ["preceding", "leader", "lead_id", "preceding_vehicle_id", "front_id"])
    lane = guess_column(cols, ["lane", "lane_id", "laneindex", "lane_kf"])
    speed = guess_column(cols, ["speed", "v", "vel", "velocity", "speed_mps", "speed_kf"])
    run_index = guess_column(cols, ["run_index", "runid", "run_id", "run", "collection_index", "collection_id"])

    xloc_col = guess_column(cols, ["xloc_kf", "x", "x_position", "xloc"])
    yloc_col = guess_column(cols, ["yloc_kf", "y", "y_position", "yloc"])

    if xloc_col and yloc_col:
        if dataset_name and ("I-90" in dataset_name or "I-94" in dataset_name or "I-294" in dataset_name):
            pos = xloc_col
            if "I-294" in dataset_name:
                print(f" Using X ({xloc_col}) as longitudinal direction (I-294 dataset)")
            else:
                print(f" Using X ({xloc_col}) as longitudinal direction (I-90/I-94 dataset)")
        elif dataset_name and "I-395" in dataset_name:
            pos = yloc_col
            print(f" Using Y ({yloc_col}) as longitudinal direction (I-395 dataset)")
        else:
            try:
                sample_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
                sample_df = sample_df.dropna(subset=[speed, xloc_col, yloc_col, time, veh_id])
                if len(sample_df) > 100:
                    # If run_index exists, ensure we don't mix trajectories from different runs.
                    group_cols = [veh_id]
                    sort_cols = [veh_id, time]
                    if run_index and run_index in sample_df.columns:
                        group_cols = [run_index, veh_id]
                        sort_cols = [run_index, veh_id, time]

                    sample_df = sample_df.sort_values(sort_cols)
                    sample_df['dt'] = sample_df.groupby(group_cols)[time].diff()
                    sample_df['dx'] = sample_df.groupby(group_cols)[xloc_col].diff().abs()
                    sample_df['dy'] = sample_df.groupby(group_cols)[yloc_col].diff().abs()
                    valid = (sample_df['dt'] > 0) & (sample_df['dt'] < 1.0)
                    if valid.sum() > 50:
                        dx_dt = (sample_df.loc[valid, 'dx'] / sample_df.loc[valid, 'dt']).replace([np.inf, -np.inf], np.nan)
                        dy_dt = (sample_df.loc[valid, 'dy'] / sample_df.loc[valid, 'dt']).replace([np.inf, -np.inf], np.nan)
                        speed_vals = sample_df.loc[valid, speed]
                        mask = ~(np.isnan(dx_dt) | np.isnan(dy_dt) | np.isnan(speed_vals))
                        if mask.sum() > 50:
                            dx_dt_clean = dx_dt[mask]
                            dy_dt_clean = dy_dt[mask]
                            speed_clean = speed_vals[mask]
                            x_avg = dx_dt_clean.mean()
                            y_avg = dy_dt_clean.mean()
                            speed_avg = speed_clean.mean()
                            x_diff = abs(x_avg - speed_avg) if x_avg > 0 else float('inf')
                            y_diff = abs(y_avg - speed_avg) if y_avg > 0 else float('inf')
                            if x_diff < y_diff:
                                pos = xloc_col
                                print(f" Detected: X ({xloc_col}) is the longitudinal direction")
                            else:
                                pos = yloc_col
                                print(f" Detected: Y ({yloc_col}) is the longitudinal direction")
                        else:
                            pos = yloc_col
                            print(f" Using Y ({yloc_col}) as longitudinal (default, insufficient data)")
                    else:
                        pos = yloc_col
                        print(f" Using Y ({yloc_col}) as longitudinal (default, insufficient data)")
                else:
                    pos = yloc_col
                    print(f" Using Y ({yloc_col}) as longitudinal (default, insufficient data)")
            except Exception as e:
                pos = yloc_col
                print(f" Using Y ({yloc_col}) as longitudinal (default, detection failed: {e})")
    elif yloc_col:
        pos = yloc_col
    elif xloc_col:
        pos = xloc_col
    else:
        pos = guess_column(cols, ["y", "y_position", "long", "longitudinal", "pos", "position", "x", "x_position"])

    veh_type = guess_column(cols, ["type", "vehicle_type", "veh_type", "class", "vehclass", "type_most_common"])

    # AV column: exact match first, with validation
    # Note: I-395 uses vehicle type=4 for AVs, not a separate AV column
    # Skip AV column detection for I-395 dataset
    av_column = None
    if dataset_name and "I-395" in dataset_name:
        # I-395 doesn't have a separate AV column - AVs are identified by type=4
        av_column = None
    else:
        av_candidates = ["av", "autonomous", "is_av", "av_flag"]
        # "acc" is excluded from exact match to avoid matching "acceleration_kf"
        cols_lower = {c.lower(): c for c in cols}
        for cand in av_candidates:
            if cand.lower() in cols_lower:
                candidate_col = cols_lower[cand.lower()]
                # Validate: check if column contains Yes/No/True/False strings
                try:
                    sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
                    if candidate_col in sample_df.columns:
                        sample_values = sample_df[candidate_col].dropna().astype(str).str.lower().str.strip()
                        if len(sample_values) > 0:
                            av_strings = ['yes', 'no', 'true', 'false', '1', '0', 'y', 'n']
                            av_count = sample_values.isin(av_strings).sum()
                            av_ratio = av_count / len(sample_values) if len(sample_values) > 0 else 0.0
                            # If less than 80% are Yes/No strings, it's probably not an AV column
                            if av_ratio >= 0.8:
                                av_column = candidate_col
                                break
                except Exception:
                    pass
        
        # If no exact match, try "acc" but with strict validation
        if av_column is None:
            if "acc" in cols_lower:
                candidate_col = cols_lower["acc"]
                # Validate that "acc" column contains Yes/No strings, not numeric acceleration
                try:
                    sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
                    if candidate_col in sample_df.columns:
                        sample_values = sample_df[candidate_col].dropna().astype(str).str.lower().str.strip()
                        if len(sample_values) > 0:
                            av_strings = ['yes', 'no', 'true', 'false', '1', '0', 'y', 'n']
                            av_count = sample_values.isin(av_strings).sum()
                            av_ratio = av_count / len(sample_values) if len(sample_values) > 0 else 0.0
                            if av_ratio >= 0.8:
                                av_column = candidate_col
                except Exception:
                    pass
        
        # Last resort: try guess_column but skip "acc" to avoid matching "acceleration_kf"
        if av_column is None:
            av_candidates_no_acc = ["av", "autonomous", "is_av", "av_flag"]
            av_column = guess_column(cols, av_candidates_no_acc)

    length = guess_column(cols, ["length", "veh_length", "vehicle_length", "length_smoothed"])

    # Infer longitudinal direction: does position increase in direction of travel?
    # Correlation of d(pos)/dt with speed: positive => pos increases downstream (leader has larger pos)
    pos_increases_downstream = True
    try:
        sample_df = df.sample(min(5000, len(df))) if len(df) > 5000 else df
        sample_df = sample_df.dropna(subset=[time, pos, speed, veh_id])
        if run_index and run_index in sample_df.columns:
            sample_df = sample_df.dropna(subset=[run_index])
        group_cols = [veh_id] if not (run_index and run_index in sample_df.columns) else [run_index, veh_id]
        sort_cols = [veh_id, time] if not (run_index and run_index in sample_df.columns) else [run_index, veh_id, time]
        sample_df = sample_df.sort_values(sort_cols)
        dpos = sample_df.groupby(group_cols)[pos].diff()
        dt = sample_df.groupby(group_cols)[time].diff()
        speed_vals = sample_df[speed].astype(float)
        valid = (dt.notna()) & (dt > 0) & (dt < 2.0) & (dpos.notna()) & (speed_vals.notna())
        if valid.sum() > 100:
            dpos_dt = (dpos / dt).replace([np.inf, -np.inf], np.nan)
            mask = valid & dpos_dt.notna()
            if mask.sum() > 100:
                corr = np.corrcoef(dpos_dt[mask].values.astype(float), speed_vals[mask].values.astype(float))[0, 1]
                if not np.isnan(corr):
                    pos_increases_downstream = corr >= 0
                    print(
                        f" Longitudinal direction: pos "
                        f"{'increases' if pos_increases_downstream else 'decreases'} downstream "
                        f"(corr d(pos)/dt vs speed = {corr:.3f})"
                    )
    except Exception as e:
        print(f" Could not infer longitudinal direction, assuming pos increases downstream ({e})")

    missing = [k for k, v in {
        "time": time, "veh_id": veh_id, "lane": lane, "speed": speed, "pos": pos, "veh_type": veh_type
    }.items() if v is None]
    if missing:
        raise ValueError(
            f"Could not infer required columns: {missing}. "
            f"Please set them manually in infer_schema(). Found columns: {cols}"
        )

    return Schema(
        time=time,
        veh_id=veh_id,
        lead_id=lead_id,
        lane=lane,
        speed=speed,
        pos=pos,
        veh_type=veh_type,
        run_index=run_index,
        av_column=av_column,
        length=length,
        pos_increases_downstream=pos_increases_downstream
    )


def compute_leader_ids(df: pd.DataFrame, sc: Schema) -> pd.DataFrame:
    """
    Compute leader IDs by picking the next vehicle ahead in same lane for each timestamp.
    Assumes df contains multiple vehicles at same time and lane.
    """
    df = df.copy()
    print(" Preparing data for leader ID computation...")

    df[sc.time] = pd.to_numeric(df[sc.time], errors="coerce")
    df[sc.veh_id] = pd.to_numeric(df[sc.veh_id], errors="coerce")
    df[sc.lane] = pd.to_numeric(df[sc.lane], errors="coerce")
    df[sc.pos] = pd.to_numeric(df[sc.pos], errors="coerce")
    # Only drop rows missing essential trajectory columns (never run_index — optional for grouping)
    drop_cols = [sc.time, sc.veh_id, sc.lane, sc.pos]
    df = df.dropna(subset=drop_cols)
    # Coerce run_index for grouping; fill NaN so grouping still works
    if sc.run_index is not None and sc.run_index in df.columns:
        df[sc.run_index] = pd.to_numeric(df[sc.run_index], errors="coerce")
        df[sc.run_index] = df[sc.run_index].fillna(-1).astype("int64")

    print(" Computing leader IDs using optimized numpy operations...")
    # Sort so that "next row" is the vehicle ahead in the direction of travel.
    # pos_increases_downstream: ahead = larger pos; else ahead = smaller pos.
    sort_cols = [sc.time, sc.lane, sc.pos]
    if sc.run_index is not None and sc.run_index in df.columns:
        sort_cols = [sc.run_index] + sort_cols
    ascending = [True] * (len(sort_cols) - 1) + [sc.pos_increases_downstream]
    df = df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    time_arr = df[sc.time].values
    lane_arr = df[sc.lane].values
    pos_arr = df[sc.pos].values
    veh_id_arr = df[sc.veh_id].values
    if sc.run_index is not None and sc.run_index in df.columns:
        run_arr = df[sc.run_index].values
    else:
        run_arr = None

    lead_id_arr = np.full(len(df), np.nan, dtype=float)

    time_changed = np.concatenate([[True], time_arr[1:] != time_arr[:-1]])
    lane_changed = np.concatenate([[True], lane_arr[1:] != lane_arr[:-1]])
    if run_arr is not None:
        run_changed = np.concatenate([[True], run_arr[1:] != run_arr[:-1]])
        group_boundary = time_changed | lane_changed | run_changed
    else:
        group_boundary = time_changed | lane_changed
    group_ids = np.cumsum(group_boundary)

    next_group = np.concatenate([group_ids[1:], [group_ids[-1] + 1]])
    next_pos = np.concatenate([pos_arr[1:], [np.nan]])
    next_veh_id = np.concatenate([veh_id_arr[1:], [np.nan]])

    same_group = (group_ids == next_group)
    ahead = (next_pos > pos_arr) if sc.pos_increases_downstream else (next_pos < pos_arr)
    valid = ~np.isnan(next_veh_id)

    mask = same_group & ahead & valid
    lead_id_arr[mask] = next_veh_id[mask]

    df["lead_id"] = lead_id_arr
    has_leader = (~np.isnan(lead_id_arr)).sum()
    print(f" Completed: {has_leader:,} vehicle-time records have identified leaders ({100*has_leader/len(df):.1f}%)")
    return df

# -----------------------------
# Prospect Theory (PT) model
# -----------------------------
# ============================================================
# PT Parameters (replace IDMParams + BOUNDS)
# PT ranges from the paper's calibrated-parameter table:
#   Wm:    (2, 8)
#   Alpha: (0, 0.6)
#   Beta:  (2, 8)
#   Wc:    (6E4, 1.3E5)
#   Tmax:  (2, 8)
#   Gamma: (0.3, 2.0)
# ============================================================

PT_EPS = 1e-9  # for stability when alpha or dv are 0

@dataclass
class PTParams:
    Wm: float      # w_m (asymmetry factor)
    Alpha: float   # α (speed uncertainty variation coefficient)
    Beta: float    # β (logit uncertainty parameter; calibrated per paper table)
    Wc: float      # W_c (accident weight)
    Tmax: float    # T_max (max anticipation time horizon)
    Gamma: float   # γ (sensitivity exponent)

PT_BOUNDS = {
    "Wm": (2.0, 8.0),
    "Alpha": (0.0, 0.6),          # paper says (0, 0.6)
    "Beta": (2.0, 8.0),
    "Wc": (6e4, 1.3e5),
    "Tmax": (2.0, 8.0),
    "Gamma": (0.3, 2.0),
}

PT_VEC_BOUNDS = np.array(
    [
        PT_BOUNDS["Wm"],
        PT_BOUNDS["Alpha"],
        PT_BOUNDS["Beta"],
        PT_BOUNDS["Wc"],
        PT_BOUNDS["Tmax"],
        PT_BOUNDS["Gamma"],
    ],
    dtype=float,
)

def pt_random_params() -> PTParams:
    return PTParams(
        Wm=random.uniform(*PT_BOUNDS["Wm"]),
        Alpha=random.uniform(*PT_BOUNDS["Alpha"]),
        Beta=random.uniform(*PT_BOUNDS["Beta"]),
        Wc=random.uniform(*PT_BOUNDS["Wc"]),
        Tmax=random.uniform(*PT_BOUNDS["Tmax"]),
        Gamma=random.uniform(*PT_BOUNDS["Gamma"]),
    )

def pt_params_to_vec(p: PTParams) -> np.ndarray:
    return np.array([p.Wm, p.Alpha, p.Beta, p.Wc, p.Tmax, p.Gamma], dtype=float)

def pt_vec_to_params(v: np.ndarray) -> PTParams:
    return PTParams(
        Wm=float(v[0]),
        Alpha=float(v[1]),
        Beta=float(v[2]),
        Wc=float(v[3]),
        Tmax=float(v[4]),
        Gamma=float(v[5]),
    )

# Normal CDF/PDF helpers (needed by PT Newton-Raphson part)
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)

# Optimized norm_cdf using scipy if available (much faster), fallback to math.erf
try:
    from scipy.special import erf as scipy_erf
    _USE_SCIPY_ERF = True
except ImportError:
    _USE_SCIPY_ERF = False

def norm_cdf(z: float) -> float:
    """Optimized normal CDF - uses scipy if available, otherwise math.erf"""
    if _USE_SCIPY_ERF:
        return 0.5 * (1.0 + scipy_erf(z / _SQRT2))
    else:
        return 0.5 * (1.0 + math.erf(z / _SQRT2))

def norm_pdf(z: float) -> float:
    return math.exp(-0.5 * z * z) / _SQRT2PI

# ============================================================
# PT Model Acceleration + Simulation (replaces IDM simulation)
# Implements equations (6)–(24).
# Notes:
# - a is clamped to [-8, 5].
# - GA calibration MUST be deterministic (no random noise) or it won't converge.
# ============================================================

PT_A_MAX = 5.0
PT_A_MIN = -8.0
PT_DT_NOISE = 0.1          # Eq (21) uses 0.1 explicitly
PT_NR_MAX_ITERS = 15  # Reduced from 20 (usually converges faster)
PT_NR_TOL = 1e-5  # Slightly relaxed from 1e-6 for faster convergence

# If the paper fixes these constants, set them here (keep them fixed, not calibrated)
PT_S0 = 1.0                # s0 (minimum gap) — keep fixed unless paper calibrates it
PT_A_MAX_FREEFLOW = 5.0    # a_max used in a_ff (Eq 22)
# v_desired: to stay paper-consistent, keep it fixed if paper gives v0.
# If paper does not provide a fixed v0, a stable deterministic proxy is:
# v_desired = percentile(follower speed, 95)

def pt_clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def pt_acceleration(
    v_f: float,
    v_l: float,
    gap: float,
    p: PTParams,
    *,
    v_desired: float,
    deterministic: bool = False,
    rng: np.random.RandomState | None = None,
) -> float:
    """
    PT acceleration computed with Newton-Raphson (Eqs 6–24 style),
    with corrected PDF usage and truly deterministic mode.
    """
    v_f = max(0.0, float(v_f))
    v_l = max(0.0, float(v_l))
    gap = max(PT_EPS, float(gap))
    dv = v_f - v_l  # Δv

    # Eq (6): S_eff = max(gap - S0, 0.1)
    S_eff = max(gap - PT_S0, 0.1)

    alpha = float(p.Alpha)
    Tmax = float(p.Tmax)
    Wm = float(p.Wm)
    Wc = float(p.Wc)
    gamma = float(p.Gamma)

    # stability
    if abs(dv) < PT_EPS:
        dv_safe = PT_EPS if dv >= 0 else -PT_EPS
    else:
        dv_safe = dv

    # Eq (7)-(8): tau
    if dv > (S_eff / max(Tmax, PT_EPS)):
        tau = S_eff / max(dv_safe, PT_EPS)
    else:
        tau = Tmax
    tau = max(tau, PT_EPS)

    if alpha <= 0.0:
        alpha = PT_EPS

    # Z' = dZ/dA = 0.5*tau/(alpha*dv)
    Zprime = tau / (2.0 * alpha * dv_safe)

    # Z* (your paper text is messy "ln ln", so keep your guard but stable)
    log_arg = Wc * Zprime
    if log_arg > PT_EPS:
        log_term = math.log(log_arg)
        if log_term >= 0.0:
            Zstar = -math.sqrt(2.0 * log_term) / _SQRT2PI
        else:
            Zstar = 0.0
    else:
        Zstar = 0.0

    # initial A*
    Astar = (2.0 / tau) * (S_eff / tau - dv + alpha * dv_safe * Zstar)

    for _ in range(PT_NR_MAX_ITERS):
        # Z(A)
        Z = (dv + 0.5 * Astar * tau - (S_eff / tau)) / (alpha * dv_safe)

        # IMPORTANT: use PDF, not CDF
        phi = norm_pdf(Z)

        # U' and U''
        if Astar >= 0.0:
            a_pos = max(Astar, PT_EPS)
            U1 = gamma * (a_pos ** (gamma - 1.0))
            U2 = gamma * (gamma - 1.0) * (a_pos ** (gamma - 2.0))
        else:
            a_neg = max(-Astar, PT_EPS)
            U1 = Wm * gamma * (a_neg ** (gamma - 1.0))
            U2 = -Wm * gamma * (gamma - 1.0) * (a_neg ** (gamma - 2.0))

        # F(A) = U' - Wc * phi(Z) * Z'
        F = U1 - Wc * phi * Zprime
        if abs(F) < PT_NR_TOL:
            break

        # d/dA[phi(Z)] = phi'(Z)*dZ/dA = (-Z*phi(Z))*Z'
        # so d/dA[ Wc*phi(Z)*Z' ] = Wc * (-Z*phi(Z))*Z'*Z'
        # => F' = U'' - [ Wc * (-Z*phi)* (Z')^2 ] = U'' + Wc*Z*phi*(Z')^2
        Fp = U2 + Wc * Z * phi * (Zprime ** 2)

        if abs(Fp) < 1e-12:
            break

        Anew = Astar - F / Fp
        if abs(Anew - Astar) < PT_NR_TOL:
            Astar = Anew
            break
        Astar = Anew

    # ---- stochastic term (TRULY off in deterministic mode) ----
    if deterministic:
        a_cf = Astar
    else:
        # If you want Beta to matter, a simple, common mapping is:
        # sigma = 1 / Beta  (bigger Beta => less randomness)
        beta = max(float(p.Beta), PT_EPS)
        sigma = 1.0 / beta

        if rng is None:
            rng = np.random.RandomState(0)
        a_cf = Astar + sigma * float(rng.normal(0.0, 1.0))

    # free-flow
    v_des = max(float(v_desired), PT_EPS)
    a_ff = PT_A_MAX_FREEFLOW * (1.0 - v_f / v_des)

    a = min(a_cf, a_ff)
    a = pt_clamp(a, PT_A_MIN, PT_A_MAX)
    return float(a)

def simulate_follower_pt(
    t: np.ndarray,
    x_lead: np.ndarray,
    v_lead: np.ndarray,
    x0: float,
    v0: float,
    p: PTParams,
    *,
    v_desired: float,
    lead_length: float = 0.0,
    follower_length: float = 0.0,
    deterministic: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    dt_arr = np.diff(t)
    dt_arr = np.clip(dt_arr, DT_MIN, DT_MAX)

    n = len(t)
    x = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    x[0] = float(x0)
    v[0] = max(0.0, float(v0))

    rng = np.random.RandomState(0)

    for i in range(n - 1):
        # compute simulated bumper-to-bumper gap from simulated x and observed leader x
        gap_sim = (x_lead[i] - x[i]) - 0.5 * lead_length - 0.5 * follower_length
        gap_sim = max(gap_sim, PT_EPS)

        a_i = pt_acceleration(
            v_f=v[i],
            v_l=v_lead[i],
            gap=gap_sim,
            p=p,
            v_desired=v_desired,
            deterministic=deterministic,
            rng=rng,
        )

        dt = dt_arr[i]

        # slightly better integrator than "x += v_next*dt"
        v_next = max(0.0, v[i] + a_i * dt)
        x_next = x[i] + v[i] * dt + 0.5 * a_i * dt * dt

        v[i + 1] = v_next
        x[i + 1] = x_next

    return x, v

# -----------------------------
# Episode extraction
# -----------------------------
@dataclass
class Episode:
    run_index: Optional[int]
    follower_id: int
    leader_id: int
    follower_type: str
    start_t: float
    end_t: float
    df: pd.DataFrame

def build_episodes(df: pd.DataFrame, sc: Schema) -> List[Episode]:
    df = df.copy()

    # leader IDs
    if sc.lead_id is None:
        print(" Leader ID column not found - computing from trajectory data...")
        df = compute_leader_ids(df, sc)
        sc.lead_id = "lead_id"
    else:
        print(" Using existing leader ID column from dataset")

    print(" Converting data types and filtering...")
    df[sc.time] = pd.to_numeric(df[sc.time], errors="coerce")
    df[sc.veh_id] = pd.to_numeric(df[sc.veh_id], errors="coerce").astype("Int64")
    df[sc.lead_id] = pd.to_numeric(df[sc.lead_id], errors="coerce").astype("Int64")
    df[sc.lane] = pd.to_numeric(df[sc.lane], errors="coerce").astype("Int64")
    df[sc.speed] = pd.to_numeric(df[sc.speed], errors="coerce")
    df[sc.pos] = pd.to_numeric(df[sc.pos], errors="coerce")
    if sc.run_index is not None and sc.run_index in df.columns:
        # run_index: coerce for grouping, fill NaN with sentinel (do NOT require in dropna)
        df[sc.run_index] = pd.to_numeric(df[sc.run_index], errors="coerce")
        df[sc.run_index] = df[sc.run_index].fillna(-1).astype("int64")

    # Vehicle type: keep raw values (string labels like "small-vehicle"/"large-vehicle" exist in TGSIM).
    # Do NOT coerce to numeric here; labeling logic below handles both numeric codes and string labels.

    # AV column: never coerce to numeric; keep raw values and parse only when labeling episodes

    # Only require essential trajectory columns (never run_index, veh_type, or av_column).
    # Keep consistent with IDM: require leader_id too.
    required = [sc.time, sc.veh_id, sc.lead_id, sc.lane, sc.speed, sc.pos]
    if sc.length and sc.length in df.columns:
        required.append(sc.length)
    initial_rows = len(df)
    df = df.dropna(subset=required)
    print(f" After removing missing values: {len(df):,} rows (removed {initial_rows - len(df):,})")
    sort_cols = [sc.veh_id, sc.time]
    if sc.run_index is not None and sc.run_index in df.columns:
        sort_cols = [sc.run_index] + sort_cols
    df = df.sort_values(sort_cols)

    # Build leader dataframe with position, speed, and length if available
    leader_cols = [sc.time, sc.veh_id, sc.pos, sc.speed]
    if sc.run_index is not None and sc.run_index in df.columns:
        leader_cols.append(sc.run_index)
    if sc.length:
        leader_cols.append(sc.length)
    
    leaders = df[leader_cols].rename(columns={
        sc.time: "t",
        sc.veh_id: "leader_id",
        sc.pos: "x_lead",
        sc.speed: "v_lead",
    })
    if sc.run_index is not None and sc.run_index in leaders.columns:
        leaders = leaders.rename(columns={sc.run_index: "run_index"})
    # Rename length column if it exists
    if sc.length:
        leaders = leaders.rename(columns={sc.length: "lead_length"})
    else:
        leaders["lead_length"] = 0.0  # Default to 0 if length not available

    follower_cols = {
        sc.veh_id: "follower_id",
        sc.lead_id: "leader_id",
        sc.pos: "x_foll",
        sc.speed: "v_foll",
        sc.lane: "lane_id",
        sc.veh_type: "veh_type_code",
        sc.time: "t",
    }
    if sc.run_index is not None and sc.run_index in df.columns:
        follower_cols[sc.run_index] = "run_index"

    if sc.av_column and sc.av_column in df.columns:
        follower_cols[sc.av_column] = sc.av_column
    
    # Add length column if available (for follower)
    if sc.length:
        follower_cols[sc.length] = sc.length

    follower = df.rename(columns=follower_cols)
    
    # Rename follower length column if it exists
    if sc.length and sc.length in follower.columns:
        follower = follower.rename(columns={sc.length: "follower_length"})
    else:
        follower["follower_length"] = 0.0  # Default to 0 if length not available

    print(" Merging follower and leader trajectories...")
    merge_keys = ["t", "leader_id"]
    if "run_index" in follower.columns and "run_index" in leaders.columns:
        merge_keys = ["run_index"] + merge_keys
    merged = follower.merge(leaders, on=merge_keys, how="inner")
    print(f" Merged data: {len(merged):,} follower-leader pairs")

    # CRITICAL FIX: Compute bumper-to-bumper gap (consistent with fitness/metrics)
    # Ensure lead_length and follower_length are numeric
    if "lead_length" not in merged.columns:
        merged["lead_length"] = 0.0
    merged["lead_length"] = pd.to_numeric(merged["lead_length"], errors="coerce").fillna(0.0)
    
    if "follower_length" not in merged.columns:
        merged["follower_length"] = 0.0
    merged["follower_length"] = pd.to_numeric(merged["follower_length"], errors="coerce").fillna(0.0)
    
    # Bumper-to-bumper gap (center-based positions):
    # gap = x_lead - x_foll - (L_lead/2) - (L_foll/2)
    # Leader rear bumper = x_lead - L_lead/2
    # Follower front bumper = x_foll + L_foll/2
    # Gap = (x_lead - L_lead/2) - (x_foll + L_foll/2) = x_lead - x_foll - L_lead/2 - L_foll/2
    merged["gap"] = merged["x_lead"] - merged["x_foll"] - merged["lead_length"]/2 - merged["follower_length"]/2

    before_gap_filter = len(merged)
    merged = merged[(merged["gap"] > 0.0) & (merged["gap"] < MAX_HEADWAY_M)]
    print(f" After gap filter (0 < gap < {MAX_HEADWAY_M}m): {len(merged):,} pairs (removed {before_gap_filter - len(merged):,})")

    print(" Extracting episodes with constraints:")
    print(f" - Duration > {MIN_EPISODE_DURATION_S}s")
    print(f" - No lane changes")
    print(f" - Constant leader ID")

    episodes: List[Episode] = []

    if "run_index" in merged.columns:
        unique_followers = merged[["run_index", "follower_id"]].drop_duplicates().to_records(index=False)
    else:
        unique_followers = merged["follower_id"].unique()
    total_followers = len(unique_followers)

    # Avoid pandas behavior differences: when grouping by a single key, use the string key
    group_keys: str | List[str]
    if "run_index" in merged.columns:
        group_keys = ["run_index", "follower_id"]
    else:
        group_keys = "follower_id"

    for idx_f, (key, g) in enumerate(merged.groupby(group_keys, sort=False), 1):
        g = g.sort_values("t").reset_index(drop=True)
        if isinstance(key, tuple):
            if len(key) == 2:
                run_i, fid = key
                run_i = int(run_i) if pd.notna(run_i) else None
            else:
                run_i, fid = None, key[0]
        else:
            run_i, fid = None, key

        leader_change = g["leader_id"].ne(g["leader_id"].shift(1))
        lane_change = g["lane_id"].ne(g["lane_id"].shift(1))
        dt = g["t"].diff()
        time_break = (dt.isna()) | (dt > DT_MAX * 2.5)

        cut = leader_change | lane_change | time_break
        seg_id = cut.cumsum()

        for _, seg in g.groupby(seg_id):
            if len(seg) < 5:
                continue

            dur = float(seg["t"].iloc[-1] - seg["t"].iloc[0])
            if dur <= MIN_EPISODE_DURATION_S:
                continue

            if seg["leader_id"].nunique() != 1:
                continue
            if seg["lane_id"].nunique() != 1:
                continue

            vt_raw = seg["veh_type_code"].iloc[0]

            # detect AV using separate column if exists (acc/av)
            is_av = False
            if sc.av_column and sc.av_column in seg.columns:
                try:
                    av_value = str(seg[sc.av_column].iloc[0]).lower().strip()
                    # Match IDM logic (some datasets store categorical strings, not just yes/no)
                    is_av = av_value in ['yes', 'true', '1', 'y', 'acc', 'av', 'autonomous']
                except Exception:
                    is_av = False

            if is_av:
                follower_type = "av"
            else:
                # Robust parsing:
                # - Some datasets store type codes as strings like "1", "2", "3", "4" (or even "1.0").
                # - Some store textual labels like "small-vehicle"/"large-vehicle".
                follower_type = "unknown"
                if not pd.isna(vt_raw):
                    # First try numeric mapping (works for int, float, numeric strings)
                    try:
                        vt = int(float(vt_raw))
                        follower_type = VEHICLE_TYPE_MAP.get(vt, "unknown")
                    except Exception:
                        # Fallback: textual labels
                        vt_lower = str(vt_raw).lower()
                        if "small" in vt_lower:
                            follower_type = "small"
                        elif "large" in vt_lower:
                            follower_type = "large"

            episodes.append(Episode(
                run_index=run_i,
                follower_id=int(fid),
                leader_id=int(seg["leader_id"].iloc[0]),
                follower_type=follower_type,
                start_t=float(seg["t"].iloc[0]),
                end_t=float(seg["t"].iloc[-1]),
                df=seg.copy()
            ))

        if idx_f % max(1, total_followers // 10) == 0 or idx_f == total_followers:
            print(f" Processed {idx_f}/{total_followers} followers, found {len(episodes)} episodes so far...")

    print(f" Episode extraction complete: {len(episodes):,} valid episodes found")
    return episodes

# -----------------------------
# Genetic Algorithm
# -----------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def fitness_episode_pt(ep: Episode, p: PTParams) -> float:
    d = ep.df
    t = d["t"].to_numpy(dtype=float)
    x_lead = d["x_lead"].to_numpy(dtype=float)
    v_lead = d["v_lead"].to_numpy(dtype=float)
    x_obs = d["x_foll"].to_numpy(dtype=float)
    v_obs = d["v_foll"].to_numpy(dtype=float)

    # Desired speed v0 (paper parameter) — if paper gives a fixed v0, use it instead.
    # Deterministic proxy (stable):
    v_desired = float(np.percentile(v_obs, 95))

    # Get vehicle lengths
    lead_length = float(d["lead_length"].iloc[0]) if "lead_length" in d.columns else 0.0
    follower_length = float(d["follower_length"].iloc[0]) if "follower_length" in d.columns else 0.0

    x_sim, v_sim = simulate_follower_pt(
        t=t,
        x_lead=x_lead,
        v_lead=v_lead,
        x0=float(x_obs[0]),
        v0=float(v_obs[0]),
        p=p,
        v_desired=v_desired,
        lead_length=lead_length,
        follower_length=follower_length,
        deterministic=True,  # must be True for GA
    )

    err = W_POS * np.abs(x_obs - x_sim) + W_SPEED * np.abs(v_obs - v_sim)
    return float(np.sum(err))

def tournament_select(pop: List[np.ndarray], fit: List[float], k: int) -> np.ndarray:
    idxs = random.sample(range(len(pop)), k)
    best = min(idxs, key=lambda i: fit[i])
    return pop[best].copy()

def crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.rand(a.size) < 0.5
    c1 = np.where(mask, a, b)
    c2 = np.where(mask, b, a)
    return c1, c2

def pt_mutate(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    for i in range(out.size):
        if random.random() < GA_MUTATION_PROB:
            lo, hi = PT_VEC_BOUNDS[i]
            span = hi - lo
            out[i] += random.gauss(0.0, GA_MUTATION_SCALE * span)
            out[i] = clamp(out[i], lo, hi)
    return out

def pt_crossover(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mask = np.random.rand(a.size) < 0.5
    c1 = np.where(mask, a, b)
    c2 = np.where(mask, b, a)
    return c1, c2

def calibrate_episode_ga_pt(ep: Episode, show_progress: bool = True) -> Tuple[PTParams, float]:
    pop = [pt_params_to_vec(pt_random_params()) for _ in range(GA_POP)]
    fit = [fitness_episode_pt(ep, pt_vec_to_params(ind)) for ind in pop]
    elite_n = max(1, int(GA_ELITE_FRAC * GA_POP))

    best_hist = []
    no_improve = 0

    for gen in range(GA_GENS):
        order = np.argsort(fit)
        pop = [pop[i] for i in order]
        fit = [fit[i] for i in order]

        best = fit[0]
        best_hist.append(best)

        if show_progress and (gen == 0 or gen % 5 == 0 or gen == GA_GENS - 1):
            print(".", end="", flush=True)

        if gen > 0:
            if (best_hist[-2] - best) < GA_EARLY_STOP_TOL:
                no_improve += 1
            else:
                no_improve = 0
            if no_improve >= GA_EARLY_STOP_GENS:
                break

        new_pop = pop[:elite_n]
        while len(new_pop) < GA_POP:
            p1 = tournament_select(pop, fit, GA_TOURN_K)
            p2 = tournament_select(pop, fit, GA_TOURN_K)
            if random.random() < GA_CROSSOVER_PROB:
                c1, c2 = pt_crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            c1 = pt_mutate(c1)
            c2 = pt_mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < GA_POP:
                new_pop.append(c2)

        pop = new_pop
        fit = [fitness_episode_pt(ep, pt_vec_to_params(ind)) for ind in pop]

    best_i = int(np.argmin(fit))
    best_p = pt_vec_to_params(pop[best_i])
    best_f = float(fit[best_i])
    return best_p, best_f

def calibrate_episode_robust_pt(ep: Episode, n_runs: int = 1, use_best: bool = True, base_seed: int = None) -> Tuple[PTParams, float, Dict]:
    """
    Run calibration multiple times and aggregate results.
    
    Args:
        ep: Episode to calibrate
        n_runs: Number of calibration runs
        use_best: If True, return best run (lowest fitness). If False, return mean parameters.
        base_seed: Base seed for runs (each run uses base_seed + run_number)
    
    Returns:
        Tuple of (best_params, best_fitness, stats_dict)
        stats_dict contains: mean_fitness, std_fitness, min_fitness, max_fitness, mean_params, std_params, all_fitnesses
    """
    if n_runs == 1:
        best_p, best_f = calibrate_episode_ga_pt(ep, show_progress=False)
        return best_p, best_f, {"mean_fitness": best_f, "std_fitness": 0.0, "n_runs": 1}
    
    all_params = []
    all_fitnesses = []
    
    for run_idx in range(n_runs):
        # Set seed for this run (different seed per run)
        if base_seed is not None:
            run_seed = base_seed + run_idx
            random.seed(run_seed)
            np.random.seed(run_seed)
        else:
            # If no base seed, use completely random (non-reproducible)
            pass
        
        params, fitness = calibrate_episode_ga_pt(ep, show_progress=False)
        all_params.append(params)
        all_fitnesses.append(fitness)
    
    all_fitnesses = np.array(all_fitnesses)
    mean_fitness = float(np.mean(all_fitnesses))
    std_fitness = float(np.std(all_fitnesses))
    min_fitness = float(np.min(all_fitnesses))
    max_fitness = float(np.max(all_fitnesses))
    
    # Calculate mean parameters (needed for stats dictionary regardless of use_best)
    mean_Wm = float(np.mean([p.Wm for p in all_params]))
    mean_Alpha = float(np.mean([p.Alpha for p in all_params]))
    mean_Beta = float(np.mean([p.Beta for p in all_params]))
    mean_Wc = float(np.mean([p.Wc for p in all_params]))
    mean_Tmax = float(np.mean([p.Tmax for p in all_params]))
    mean_Gamma = float(np.mean([p.Gamma for p in all_params]))
    
    if use_best:
        # Return parameters from the run with lowest fitness
        best_idx = int(np.argmin(all_fitnesses))
        best_p = all_params[best_idx]
        best_f = all_fitnesses[best_idx]
    else:
        # Return mean parameters across all runs
        best_p = PTParams(Wm=mean_Wm, Alpha=mean_Alpha, Beta=mean_Beta, Wc=mean_Wc, Tmax=mean_Tmax, Gamma=mean_Gamma)
        best_f = mean_fitness
    
    # Calculate parameter statistics
    std_Wm = float(np.std([p.Wm for p in all_params]))
    std_Alpha = float(np.std([p.Alpha for p in all_params]))
    std_Beta = float(np.std([p.Beta for p in all_params]))
    std_Wc = float(np.std([p.Wc for p in all_params]))
    std_Tmax = float(np.std([p.Tmax for p in all_params]))
    std_Gamma = float(np.std([p.Gamma for p in all_params]))
    
    stats = {
        "n_runs": n_runs,
        "mean_fitness": mean_fitness,
        "std_fitness": std_fitness,
        "min_fitness": min_fitness,
        "max_fitness": max_fitness,
        "mean_params": {
            "Wm": mean_Wm, "Alpha": mean_Alpha, "Beta": mean_Beta, "Wc": mean_Wc, "Tmax": mean_Tmax, "Gamma": mean_Gamma
        },
        "std_params": {
            "Wm": std_Wm, "Alpha": std_Alpha, "Beta": std_Beta, "Wc": std_Wc, "Tmax": std_Tmax, "Gamma": std_Gamma
        },
        "all_fitnesses": all_fitnesses.tolist()
    }
    
    return best_p, best_f, stats

# -----------------------------
# Visualization / metrics
# -----------------------------
def calculate_performance_metrics_pt(ep: Episode, params: PTParams) -> Dict[str, float]:
    d = ep.df
    t = d["t"].to_numpy(dtype=float)
    x_lead = d["x_lead"].to_numpy(dtype=float)
    v_lead = d["v_lead"].to_numpy(dtype=float)
    x_obs = d["x_foll"].to_numpy(dtype=float)
    v_obs = d["v_foll"].to_numpy(dtype=float)

    v_desired = float(np.percentile(v_obs, 95))

    # Get vehicle lengths
    lead_length = float(d["lead_length"].iloc[0]) if "lead_length" in d.columns else 0.0
    follower_length = float(d["follower_length"].iloc[0]) if "follower_length" in d.columns else 0.0

    x_sim, v_sim = simulate_follower_pt(
        t=t,
        x_lead=x_lead,
        v_lead=v_lead,
        x0=float(x_obs[0]),
        v0=float(v_obs[0]),
        p=params,
        v_desired=v_desired,
        lead_length=lead_length,
        follower_length=follower_length,
        deterministic=True,
    )

    pos_errors = x_obs - x_sim
    rmse = np.sqrt(np.mean(pos_errors ** 2))
    mae = np.mean(np.abs(pos_errors))

    ss_res = np.sum((x_obs - x_sim) ** 2)
    ss_tot = np.sum((x_obs - np.mean(x_obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": float(rmse), "mae": float(mae), "r_squared": float(r_squared)}

def plot_episode_comparison(ep: Episode, params: PTParams, output_path: str):
    if not MATPLOTLIB_AVAILABLE:
        print(f" Warning: matplotlib not available, skipping plot for episode {ep.follower_id}")
        return

    d = ep.df
    t = d["t"].to_numpy(dtype=float)
    x_obs = d["x_foll"].to_numpy(dtype=float)
    v_obs = d["v_foll"].to_numpy(dtype=float)
    x_lead = d["x_lead"].to_numpy(dtype=float)
    v_lead = d["v_lead"].to_numpy(dtype=float)
    gap = d["gap"].to_numpy(dtype=float)
    
    # Get vehicle lengths for gap calculation
    lead_length = float(d["lead_length"].iloc[0]) if "lead_length" in d.columns else 0.0
    follower_length = float(d["follower_length"].iloc[0]) if "follower_length" in d.columns else 0.0

    v_desired = float(np.percentile(v_obs, 95))

    x_sim, v_sim = simulate_follower_pt(
        t=t,
        x_lead=x_lead,
        v_lead=v_lead,
        x0=float(x_obs[0]),
        v0=float(v_obs[0]),
        p=params,
        v_desired=v_desired,
        lead_length=lead_length,
        follower_length=follower_length,
        deterministic=True,
    )
    
    # Calculate simulated gap (center-based positions)
    gap_sim = x_lead - x_sim - lead_length/2 - follower_length/2

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    type_label = ep.follower_type.upper() if ep.follower_type != "av" else "AV"
    fig.suptitle(
        f"Episode: Follower {ep.follower_id} following Leader {ep.leader_id}\n"
        f"Vehicle Type: {type_label}, Duration: {ep.end_t - ep.start_t:.1f}s\n"
        f"PT Params: Wm={params.Wm:.2f}, Alpha={params.Alpha:.2f}, Beta={params.Beta:.2f}, "
        f"Wc={params.Wc:.0f}, Tmax={params.Tmax:.2f}, Gamma={params.Gamma:.2f}",
        fontsize=11
    )

    axes[0].plot(t, x_obs, label="Observed", linewidth=2)
    axes[0].plot(t, x_sim, "--", label="Simulated (PT)", linewidth=2)
    axes[0].plot(t, x_lead, ":", label="Leader (Observed)", linewidth=2, alpha=0.9)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Longitudinal Position")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, v_obs, label="Observed", linewidth=2)
    axes[1].plot(t, v_sim, "--", label="Simulated (PT)", linewidth=2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_title("Speed")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, gap, label="Observed Gap", linewidth=2)
    axes[2].plot(t, gap_sim, "--", label="Simulated Gap (PT)", linewidth=2)
    axes[2].axhline(y=MAX_HEADWAY_M, linestyle=":", label=f"Max headway ({MAX_HEADWAY_M}m)", alpha=0.5)
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Gap (m)")
    axes[2].set_title("Gap to Leader")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

# -----------------------------
# Tables
# -----------------------------
def create_parameters_table_pt(res_df: pd.DataFrame) -> pd.DataFrame:
    param_summary = (
        res_df[res_df["follower_type"].isin(["small", "large", "av"])]
        .groupby("follower_type")
        .agg({
            "Wm": "mean", "Alpha": "mean", "Beta": "mean",
            "Wc": "mean", "Tmax": "mean", "Gamma": "mean",
            "follower_type": "count"
        })
        .rename(columns={"follower_type": "count"})
    )

    # order matches paper's PT table: Wm, Alpha, Beta, Wc, Tmax, Gamma
    params_info = [
        ("Wm",   "Wm (Asymmetry Factor)", "(2, 8)"),
        ("Alpha","Alpha (Speed Uncertainty)", "(0, 0.6)"),
        ("Beta", "Beta (Logit Uncertainty)", "(2, 8)"),
        ("Wc",   "Wc (Accident Weight)", "(6E4, 1.3E5)"),
        ("Tmax", "Tmax (Max Horizon)", "(2, 8.0)"),
        ("Gamma","Gamma (Sensitivity Exponent)", "(0.3, 2.0)"),
    ]

    table_data = []
    for key, name, rng in params_info:
        table_data.append({
            "Model": "PT",
            "Parameter": name,
            "Range": rng,
            "Small Vehicles": f"{param_summary.loc['small', key]:.3f}" if 'small' in param_summary.index else "N/A",
            "Large Vehicles": f"{param_summary.loc['large', key]:.3f}" if 'large' in param_summary.index else "N/A",
            "Autonomous Vehicles": f"{param_summary.loc['av', key]:.3f}" if 'av' in param_summary.index else "N/A",
        })

    table_data.append({
        "Model": "PT",
        "Parameter": "count",
        "Range": "-",
        "Small Vehicles": f"{int(param_summary.loc['small', 'count'])}" if 'small' in param_summary.index else "0",
        "Large Vehicles": f"{int(param_summary.loc['large', 'count'])}" if 'large' in param_summary.index else "0",
        "Autonomous Vehicles": f"{int(param_summary.loc['av', 'count'])}" if 'av' in param_summary.index else "0",
    })

    df_table = pd.DataFrame(table_data)
    return df_table[["Model", "Parameter", "Range", "Small Vehicles", "Large Vehicles", "Autonomous Vehicles"]]

def create_performance_table_pt(res_df: pd.DataFrame) -> pd.DataFrame:
    perf_summary = (
        res_df[res_df["follower_type"].isin(["small", "large", "av"])]
        .groupby("follower_type")
        .agg({"rmse": "mean", "mae": "mean", "r_squared": "mean"})
    )

    table_data = []
    for vtype in ["small", "large", "av"]:
        if vtype in perf_summary.index:
            type_label = "Small Vehicles" if vtype == "small" else ("Large Vehicles" if vtype == "large" else "Autonomous Vehicles")
            table_data.append({
                "Model": "PT",
                "Vehicle Type": type_label,
                "RMSE": f"{perf_summary.loc[vtype, 'rmse']:.3f}",
                "MAE": f"{perf_summary.loc[vtype, 'mae']:.3f}",
                "R-squared": f"{perf_summary.loc[vtype, 'r_squared']:.4f}",
            })

    return pd.DataFrame(table_data)

def print_formatted_table(df: pd.DataFrame, title: str):
    col_widths = {}
    for col in df.columns:
        col_widths[col] = max(len(str(col)), df[col].astype(str).str.len().max())
        col_widths[col] = max(col_widths[col], 12)

    total_width = sum(col_widths.values()) + (len(df.columns) - 1) * 3 + 4
    print(f"\n{'=' * total_width}")
    print(f"{title:^{total_width}}")
    print(f"{'=' * total_width}")

    header_parts = [f"{col:^{col_widths[col]}}" for col in df.columns]
    header = " | ".join(header_parts)
    print(f"| {header} |")
    print(f"{'-' * total_width}")

    for _, row in df.iterrows():
        row_parts = [f"{str(row[col]):^{col_widths[col]}}" for col in df.columns]
        row_str = " | ".join(row_parts)
        print(f"| {row_str} |")

    print(f"{'=' * total_width}\n")

def create_episodes_excel(res_df: pd.DataFrame, output_path: str):
    """
    Create an Excel file with episode summary including leader, follower, time, and gap information.
    """
    try:
        # Select and reorder columns for Excel output
        excel_columns = [
            "dataset",
            "run_index",
            "follower_id",
            "leader_id",
            "follower_type",
            "start_t",
            "end_t",
            "duration_s",
            "min_gap",
            "max_gap",
        ]
        
        # Check which columns exist in the dataframe
        available_columns = [col for col in excel_columns if col in res_df.columns]
        excel_df = res_df[available_columns].copy()
        
        # Rename columns for better readability
        excel_df = excel_df.rename(columns={
            "dataset": "Dataset",
            "run_index": "Run Index",
            "follower_id": "Follower ID",
            "leader_id": "Leader ID",
            "follower_type": "Vehicle Type",
            "start_t": "Start Time (s)",
            "end_t": "End Time (s)",
            "duration_s": "Duration (s)",
            "min_gap": "Min Gap (m)",
            "max_gap": "Max Gap (m)",
        })
        
        # Sort by dataset, then by follower_id, then by start_t
        if "Dataset" in excel_df.columns:
            sort_cols = ["Dataset", "Follower ID", "Start Time (s)"]
            if "Run Index" in excel_df.columns:
                sort_cols = ["Dataset", "Run Index", "Follower ID", "Start Time (s)"]
            excel_df = excel_df.sort_values(sort_cols)
        
        # Write to Excel
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            excel_df.to_excel(writer, sheet_name='Episodes Summary', index=False)
            
            # Auto-adjust column widths
            from openpyxl.utils import get_column_letter
            worksheet = writer.sheets['Episodes Summary']
            for idx, col in enumerate(excel_df.columns, 1):
                max_length = max(
                    excel_df[col].astype(str).str.len().max(),
                    len(str(col))
                )
                # Set column width (add some padding, max 50)
                col_letter = get_column_letter(idx)
                worksheet.column_dimensions[col_letter].width = min(max_length + 2, 50)
        
        return True
    except ImportError:
        print(f" WARNING: openpyxl not available. Install with: pip install openpyxl")
        return False
    except Exception as e:
        print(f" WARNING: Could not create Excel file: {e}")
        return False

# -----------------------------
# Statistical tests for PT parameters (Welch ANOVA + Games–Howell + Kruskal–Wallis)
# -----------------------------
PT_PARAMS_FOR_TESTS = ["Tmax", "Alpha", "Beta", "Wc", "Wm", "Gamma"]


def _sig_label(p: float, alpha: float = 0.05) -> str:
    return "Significant Difference" if (p is not None and p < alpha) else "No Significant Difference"


def run_welch_anova_pt(res_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Welch ANOVA (3-group global test, unequal variances OK).
    One-way Welch ANOVA across follower types (small, large, av) for PT params.
    """
    try:
        import pingouin as pg
    except ImportError:
        print("ERROR: pingouin is required for Welch ANOVA. Install via: pip install pingouin")
        return pd.DataFrame()

    df = res_df.copy()
    df = df[df["follower_type"].isin(["small", "large", "av"])]

    rows = []
    for param in PT_PARAMS_FOR_TESTS:
        long_df = df[["follower_type", param]].dropna()
        long_df = long_df.rename(columns={"follower_type": "group", param: "value"})
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        long_df = long_df.dropna(subset=["value"])
        if long_df["group"].nunique() < 2 or len(long_df) < 3:
            rows.append({
                "Parameter": param,
                "F-value": np.nan,
                "p-value": np.nan,
                "Significance": "Insufficient data"
            })
            continue
        try:
            wa = pg.welch_anova(dv="value", between="group", data=long_df)
            f_val = float(wa.loc[0, "F"])
            p_val = float(wa.loc[0, "p-unc"])
            rows.append({
                "Parameter": param,
                "F-value": f_val,
                "p-value": p_val,
                "Significance": _sig_label(p_val, alpha=alpha)
            })
        except Exception as e:
            rows.append({
                "Parameter": param,
                "F-value": np.nan,
                "p-value": np.nan,
                "Significance": f"Error: {e}"
            })

    return pd.DataFrame(rows, columns=["Parameter", "F-value", "p-value", "Significance"])


def run_games_howell_pt(res_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Games–Howell post-hoc (pairwise, unequal variances OK).
    Returns pairwise comparisons with Hedges g effect size.
    """
    try:
        import pingouin as pg
    except ImportError:
        print("ERROR: pingouin is required for Games–Howell. Install via: pip install pingouin")
        return pd.DataFrame()

    df = res_df.copy()
    df = df[df["follower_type"].isin(["small", "large", "av"])]

    out = []
    for param in PT_PARAMS_FOR_TESTS:
        long_df = df[["follower_type", param]].dropna()
        long_df = long_df.rename(columns={"follower_type": "group", param: "value"})
        long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
        long_df = long_df.dropna(subset=["value"])
        if long_df["group"].nunique() < 2 or len(long_df) < 3:
            continue
        try:
            gh = pg.pairwise_gameshowell(dv="value", between="group", data=long_df)
        except Exception:
            continue
        if gh is None or len(gh) == 0:
            continue
        gh = gh.copy()
        gh["Parameter"] = param
        gh["Significance"] = gh["pval"].apply(lambda x: _sig_label(float(x), alpha=alpha))
        rename_map = {
            "A": "Group 1",
            "B": "Group 2",
            "T": "t-Statistic",
            "pval": "p-Value",
            "hedges": "Hedges g"
        }
        if "hedges" not in gh.columns:
            gh["hedges"] = np.nan
        sel = gh[["Parameter", "A", "B", "T", "df", "pval", "hedges", "Significance"]].rename(columns=rename_map)
        out.append(sel)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def run_kruskal_pt(res_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Optional robustness check:
    Kruskal–Wallis across follower types (small, large, av) for PT params.
    Nonparametric (rank-based), useful if distributions are skewed/bounded.
    """
    if not SCIPY_STATS_AVAILABLE:
        print("ERROR: scipy is required for Kruskal–Wallis. Install via: pip install scipy")
        return pd.DataFrame()

    df = res_df.copy()
    df = df[df["follower_type"].isin(["small", "large", "av"])]

    rows = []
    for param in PT_PARAMS_FOR_TESTS:
        groups = []
        ok = True
        for g in ["small", "large", "av"]:
            arr = pd.to_numeric(df.loc[df["follower_type"] == g, param], errors="coerce").dropna().to_numpy(dtype=float)
            if len(arr) < 2:
                ok = False
                break
            groups.append(arr)

        if not ok:
            rows.append({
                "Parameter": param,
                "H-stat": np.nan,
                "p-value": np.nan,
                "Significance": "Insufficient data"
            })
            continue

        h_stat, p_val = stats.kruskal(*groups)
        p_val = float(p_val)
        rows.append({
            "Parameter": param,
            "H-stat": float(h_stat),
            "p-value": p_val,
            "Significance": _sig_label(p_val, alpha=alpha)
        })

    return pd.DataFrame(rows, columns=["Parameter", "H-stat", "p-value", "Significance"])


def print_formatted_table_numeric(df: pd.DataFrame, title: str, float_cols: List[str] = None):
    """
    Like print_formatted_table, but formats selected float columns nicely.
    """
    df2 = df.copy()
    float_cols = float_cols or []
    for c in float_cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda x: f"{x:.6g}" if pd.notna(x) else "NaN")
    print_formatted_table(df2, title)

# -----------------------------
# Parallel processing worker function
# -----------------------------
def _calibrate_episode_worker(args: Tuple) -> Tuple[int, Dict, PTParams, Optional[Dict]]:
    """
    Worker function for parallel episode calibration.
    Args: (episode_idx, episode, dataset_name, base_seed, episode_base_seed)
    Returns: (episode_idx, result_dict, best_p, calib_stats)
    """
    episode_idx, ep, dataset_name, base_seed, episode_base_seed = args
    
    # Set seed for this episode if specified
    if episode_base_seed is not None:
        random.seed(episode_base_seed)
        np.random.seed(episode_base_seed)
    elif base_seed is not None:
        random.seed(base_seed + episode_idx - 1)
        np.random.seed(base_seed + episode_idx - 1)
    
    # Calibrate episode
    if N_CALIBRATION_RUNS > 1:
        best_p, best_fit, calib_stats = calibrate_episode_robust_pt(
            ep,
            n_runs=N_CALIBRATION_RUNS,
            use_best=USE_BEST_RUN,
            base_seed=episode_base_seed
        )
    else:
        best_p, best_fit = calibrate_episode_ga_pt(ep, show_progress=False)
        calib_stats = None
    
    # Calculate metrics
    metrics = calculate_performance_metrics_pt(ep, best_p)
    
    # Calculate min and max gap
    gap_values = ep.df["gap"].to_numpy(dtype=float) if "gap" in ep.df.columns else np.array([])
    min_gap = float(np.min(gap_values)) if len(gap_values) > 0 else np.nan
    max_gap = float(np.max(gap_values)) if len(gap_values) > 0 else np.nan
    
    # Build result dictionary
    result_dict = {
        "dataset": dataset_name,
        "episode_idx": episode_idx,
        "run_index": ep.run_index,
        "follower_id": ep.follower_id,
        "leader_id": ep.leader_id,
        "follower_type": ep.follower_type,
        "start_t": ep.start_t,
        "end_t": ep.end_t,
        "duration_s": ep.end_t - ep.start_t,
        "min_gap": min_gap,
        "max_gap": max_gap,
        "fitness": best_fit,
        "Wm": best_p.Wm,
        "Alpha": best_p.Alpha,
        "Beta": best_p.Beta,
        "Wc": best_p.Wc,
        "Tmax": best_p.Tmax,
        "Gamma": best_p.Gamma,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r_squared": metrics["r_squared"],
    }
    
    # Add calibration statistics if multiple runs were used
    if N_CALIBRATION_RUNS > 1 and calib_stats:
        result_dict["n_calib_runs"] = calib_stats["n_runs"]
        result_dict["fitness_mean"] = calib_stats["mean_fitness"]
        result_dict["fitness_std"] = calib_stats["std_fitness"]
        result_dict["fitness_min"] = calib_stats["min_fitness"]
        result_dict["fitness_max"] = calib_stats["max_fitness"]
        result_dict["Wm_std"] = calib_stats["std_params"]["Wm"]
        result_dict["Alpha_std"] = calib_stats["std_params"]["Alpha"]
        result_dict["Beta_std"] = calib_stats["std_params"]["Beta"]
        result_dict["Wc_std"] = calib_stats["std_params"]["Wc"]
        result_dict["Tmax_std"] = calib_stats["std_params"]["Tmax"]
        result_dict["Gamma_std"] = calib_stats["std_params"]["Gamma"]
    
    return episode_idx, result_dict, best_p, calib_stats

# -----------------------------
# Main pipeline
# -----------------------------
def process_single_dataset(csv_path: str, dataset_name: str) -> List[Dict]:
    print(f"\n{'=' * 70}")
    print(f"Processing Dataset: {dataset_name}")
    print(f"{'=' * 70}")

    print(f"\n[Step 1/7] Loading CSV file...")
    print(f" Reading from: {csv_path}")

    if not os.path.exists(csv_path):
        print(f" WARNING: File not found: {csv_path}")
        print(f" Skipping this dataset...")
        return []

    df = pd.read_csv(csv_path)
    print(f" Loaded {len(df):,} rows")

    print("\n[Step 2/7] Inferring column schema...")
    sc = infer_schema(df, dataset_name=dataset_name)
    print(f" Detected columns:")
    print(f" - Time: {sc.time}")
    print(f" - Vehicle ID: {sc.veh_id}")
    print(f" - Leader ID: {sc.lead_id if sc.lead_id else 'Will be computed'}")
    print(f" - Lane: {sc.lane}")
    print(f" - Speed: {sc.speed}")
    print(f" - Position: {sc.pos} (longitudinal direction)")
    print(f" - Vehicle Type: {sc.veh_type}")
    if sc.run_index:
        print(f" - Run Index: {sc.run_index} (separates distinct collection runs)")
    if sc.av_column:
        print(f" - AV Column: {sc.av_column}")
    if sc.length:
        print(f" - Length: {sc.length}")

    print("\n[Step 3/7] Extracting car-following episodes...")
    episodes = build_episodes(df, sc)
    print(f" Found {len(episodes):,} valid car-following episodes")

    if len(episodes) == 0:
        print("\n WARNING: No valid episodes found! Check your data and filters.")
        return []

    durations = [ep.end_t - ep.start_t for ep in episodes]
    print(f" Episode duration range: {min(durations):.1f}s - {max(durations):.1f}s")

    episodes_by_type: Dict[str, List[Episode]] = {}
    for ep in episodes:
        if ep.follower_type != "unknown":
            episodes_by_type.setdefault(ep.follower_type, []).append(ep)

    print("\n[Step 4/7] Selecting episodes for calibration...")
    for vtype, ep_list in episodes_by_type.items():
        print(f" - {vtype}: {len(ep_list)} episodes")

    if CALIBRATE_ONLY_NEAR_AVS:
        # Mode 1: Only calibrate vehicles near AVs (equal sampling)
        print("\n Mode: Calibrating only vehicles near AVs (equal sampling)")
        
        av_episodes = episodes_by_type.get("av", [])
        n_avs = len(av_episodes)

        if n_avs == 0:
            print(f"\n WARNING: NO AV EPISODES FOUND in {dataset_name}!")
            print(" ERROR: Cannot proceed with equal sampling without AVs.")
            return []

        print(f" All {n_avs} AV episodes will be used for calibration")

        # Equal sampling around AVs (time overlap +/- 5s and same lane OR within 100m)
        av_info = []
        for ep in av_episodes:
            lane_id = ep.df["lane_id"].iloc[0] if "lane_id" in ep.df.columns else None
            pos_min = ep.df["x_foll"].min() if "x_foll" in ep.df.columns else None
            pos_max = ep.df["x_foll"].max() if "x_foll" in ep.df.columns else None
            av_info.append((ep.run_index, ep.start_t, ep.end_t, ep.follower_id, lane_id, pos_min, pos_max))

        def is_around_av(ep: Episode, av_info: List[Tuple], time_tolerance: float = 5.0, spatial_tolerance: float = 100.0) -> bool:
            ep_lane = ep.df["lane_id"].iloc[0] if "lane_id" in ep.df.columns else None
            ep_pos_min = ep.df["x_foll"].min() if "x_foll" in ep.df.columns else None
            ep_pos_max = ep.df["x_foll"].max() if "x_foll" in ep.df.columns else None

            for av_run, av_start, av_end, av_id, av_lane, av_pos_min, av_pos_max in av_info:
                # Critical: must be the same data-collection run, otherwise time/space are not comparable
                if ep.run_index is not None and av_run is not None and ep.run_index != av_run:
                    continue
                time_overlap = not (ep.end_t < av_start - time_tolerance or ep.start_t > av_end + time_tolerance)
                if time_overlap:
                    same_lane = (ep_lane is not None and av_lane is not None and ep_lane == av_lane)
                    nearby_pos = False
                    if (ep_pos_min is not None and ep_pos_max is not None and av_pos_min is not None and av_pos_max is not None):
                        ep_center = (ep_pos_min + ep_pos_max) / 2
                        av_center = (av_pos_min + av_pos_max) / 2
                        nearby_pos = abs(ep_center - av_center) < spatial_tolerance
                    if same_lane or nearby_pos:
                        return True
            return False

        all_small = episodes_by_type.get("small", [])
        all_large = episodes_by_type.get("large", [])
        small_around = [ep for ep in all_small if is_around_av(ep, av_info)]
        large_around = [ep for ep in all_large if is_around_av(ep, av_info)]

        episodes_to_calibrate = []
        episodes_to_calibrate.extend(av_episodes)

        n_small_to_sample = min(n_avs, len(small_around))
        if n_small_to_sample > 0:
            sampled_small = random.sample(small_around, n_small_to_sample) if len(small_around) > n_small_to_sample else small_around
            episodes_to_calibrate.extend(sampled_small)

        n_large_to_sample = min(n_avs, len(large_around))
        if n_large_to_sample > 0:
            sampled_large = random.sample(large_around, n_large_to_sample) if len(large_around) > n_large_to_sample else large_around
            episodes_to_calibrate.extend(sampled_large)

        print(f"\n Final episodes to calibrate: {len(episodes_to_calibrate)} (AV={n_avs}, small={n_small_to_sample}, large={n_large_to_sample})")
    else:
        # Mode 2: Calibrate ALL episodes (no filtering)
        print("\n Mode: Calibrating ALL episodes (no filtering)")
        episodes_to_calibrate = episodes
        n_avs = len(episodes_by_type.get("av", []))
        n_small = len(episodes_by_type.get("small", []))
        n_large = len(episodes_by_type.get("large", []))
        print(f" Final episodes to calibrate: {len(episodes_to_calibrate)} (AV={n_avs}, small={n_small}, large={n_large})")

    print("\n[Step 4b/7] Calibrating selected episodes...")
    if NUMBA_AVAILABLE:
        print(" - Numba enabled")
    else:
        print(" - Numba not available (pip install numba for speed)")
    
    if N_CALIBRATION_RUNS > 1:
        print(f" - Using robust calibration: {N_CALIBRATION_RUNS} runs per episode ({'best run' if USE_BEST_RUN else 'mean parameters'})")
        print(f"   This will take ~{N_CALIBRATION_RUNS}x longer but provides more robust results")

    results = []
    calibrated = []
    total_episodes = len(episodes_to_calibrate)
    
    # Base seed for multiple runs (if RANDOM_SEED is set, use it as base)
    # Each episode gets: base_seed + episode_idx * N_CALIBRATION_RUNS
    # Each run within episode gets: base_seed + episode_idx * N_CALIBRATION_RUNS + run_idx
    base_seed = RANDOM_SEED if RANDOM_SEED is not None else None

    # Determine number of parallel workers for episode calibration.
    # - None => use all CPU cores
    # - 1 => no parallelization (sequential)
    # - int > 1 => multiprocessing Pool with that many processes
    if N_PARALLEL_WORKERS is None:
        n_workers = max(1, int(cpu_count()))
    else:
        try:
            n_workers = int(N_PARALLEL_WORKERS)
        except Exception:
            print(f" WARNING: Invalid N_PARALLEL_WORKERS={N_PARALLEL_WORKERS!r}; defaulting to 1 (sequential).")
            n_workers = 1
    n_workers = max(1, n_workers)
    # Don't spawn more processes than episodes to run
    n_workers = min(n_workers, total_episodes)

    if n_workers > 1:
        print(f" - Using {n_workers} parallel workers for episode calibration (expected ~{n_workers}x speedup)")
    
    # Prepare arguments for parallel processing
    worker_args = []
    for i, ep in enumerate(episodes_to_calibrate, 1):
        episode_base_seed = None
        if base_seed is not None and N_CALIBRATION_RUNS > 1:
            episode_base_seed = base_seed + (i - 1) * N_CALIBRATION_RUNS
        worker_args.append((i, ep, dataset_name, base_seed, episode_base_seed))
    
    # Process episodes (parallel or sequential)
    if n_workers > 1:
        # Parallel processing (with safe fallback for Windows low-VM situations)
        try:
            with Pool(processes=n_workers, maxtasksperchild=1) as pool:
                completed = 0
                episode_results = {}

                for result in pool.imap_unordered(_calibrate_episode_worker, worker_args):
                    episode_idx, result_dict, best_p, calib_stats = result
                    episode_results[episode_idx] = (result_dict, best_p, calib_stats)
                    completed += 1

                    if completed == 1 or completed % 10 == 0 or completed == total_episodes:
                        ep = episodes_to_calibrate[episode_idx - 1]
                        type_label = ep.follower_type.upper() if ep.follower_type != "av" else "AV"
                        print(
                            f" [{completed}/{total_episodes}] Completed {type_label} episode: "
                            f"Follower {ep.follower_id} -> Leader {ep.leader_id} "
                            f"(RMSE={result_dict['rmse']:.3f}m, R^2={result_dict['r_squared']:.4f})"
                        )

                for i in range(1, total_episodes + 1):
                    if i in episode_results:
                        result_dict, best_p, calib_stats = episode_results[i]
                        results.append(result_dict)
                        ep = episodes_to_calibrate[i - 1]
                        calibrated.append((ep, best_p, dataset_name))
        except Exception as e:
            print(f" WARNING: Parallel processing failed ({e}). Falling back to sequential calibration.")
            print("         Tip: set N_PARALLEL_WORKERS = 1 to avoid this on low-memory systems.")
            results = []
            calibrated = []
            n_workers = 1
    else:
        # Sequential processing (original code)
        for i, ep in enumerate(episodes_to_calibrate, 1):
            show_progress = (i == 1) or (i % 10 == 0) or (i == total_episodes)
            if show_progress:
                type_label = ep.follower_type.upper() if ep.follower_type != "av" else "AV"
                if N_CALIBRATION_RUNS > 1:
                    print(f" [{i}/{total_episodes}] Calibrating {type_label} episode: Follower {ep.follower_id} -> Leader {ep.leader_id} ({N_CALIBRATION_RUNS} runs)...", end="", flush=True)
                else:
                    print(f" [{i}/{total_episodes}] Calibrating {type_label} episode: Follower {ep.follower_id} -> Leader {ep.leader_id}...", end="", flush=True)

            if N_CALIBRATION_RUNS > 1:
                # Calculate episode-specific base seed for reproducibility
                episode_base_seed = None
                if base_seed is not None:
                    episode_base_seed = base_seed + (i - 1) * N_CALIBRATION_RUNS
                best_p, best_fit, calib_stats = calibrate_episode_robust_pt(
                    ep, 
                    n_runs=N_CALIBRATION_RUNS, 
                    use_best=USE_BEST_RUN,
                    base_seed=episode_base_seed
                )
            else:
                # For single run, reset seed if specified (for reproducibility)
                if base_seed is not None:
                    random.seed(base_seed + i - 1)
                    np.random.seed(base_seed + i - 1)
                best_p, best_fit = calibrate_episode_ga_pt(ep, show_progress=show_progress)
                calib_stats = None
            
            metrics = calculate_performance_metrics_pt(ep, best_p)

            if show_progress:
                if N_CALIBRATION_RUNS > 1 and calib_stats:
                    print(
                        f" Done! (RMSE={metrics['rmse']:.3f}m, R^2={metrics['r_squared']:.4f}, "
                        f"fitness={best_fit:.2f}+/-{calib_stats['std_fitness']:.2f} over {N_CALIBRATION_RUNS} runs)"
                    )
                else:
                    print(f" Done! (RMSE={metrics['rmse']:.3f}m, R^2={metrics['r_squared']:.4f})")

            # Calculate min and max gap for this episode
            gap_values = ep.df["gap"].to_numpy(dtype=float) if "gap" in ep.df.columns else np.array([])
            min_gap = float(np.min(gap_values)) if len(gap_values) > 0 else np.nan
            max_gap = float(np.max(gap_values)) if len(gap_values) > 0 else np.nan

            result_dict = {
                "dataset": dataset_name,
                "episode_idx": i,
                "run_index": ep.run_index,
                "follower_id": ep.follower_id,
                "leader_id": ep.leader_id,
                "follower_type": ep.follower_type,
                "start_t": ep.start_t,
                "end_t": ep.end_t,
                "duration_s": ep.end_t - ep.start_t,
                "min_gap": min_gap,
                "max_gap": max_gap,
                "fitness": best_fit,
                "Wm": best_p.Wm,
                "Alpha": best_p.Alpha,
                "Beta": best_p.Beta,
                "Wc": best_p.Wc,
                "Tmax": best_p.Tmax,
                "Gamma": best_p.Gamma,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r_squared": metrics["r_squared"],
            }
            
            # Add calibration statistics if multiple runs were used
            if N_CALIBRATION_RUNS > 1 and calib_stats:
                result_dict["n_calib_runs"] = calib_stats["n_runs"]
                result_dict["fitness_mean"] = calib_stats["mean_fitness"]
                result_dict["fitness_std"] = calib_stats["std_fitness"]
                result_dict["fitness_min"] = calib_stats["min_fitness"]
                result_dict["fitness_max"] = calib_stats["max_fitness"]
                result_dict["Wm_std"] = calib_stats["std_params"]["Wm"]
                result_dict["Alpha_std"] = calib_stats["std_params"]["Alpha"]
                result_dict["Beta_std"] = calib_stats["std_params"]["Beta"]
                result_dict["Wc_std"] = calib_stats["std_params"]["Wc"]
                result_dict["Tmax_std"] = calib_stats["std_params"]["Tmax"]
                result_dict["Gamma_std"] = calib_stats["std_params"]["Gamma"]
            
            results.append(result_dict)
            calibrated.append((ep, best_p, dataset_name))

    if PLOT_COMPARISONS and MATPLOTLIB_AVAILABLE:
        plots_dir = os.path.join(RESULTS_DIR, "comparison_plots", dataset_name)
        os.makedirs(plots_dir, exist_ok=True)

        total_episodes = len(calibrated)
        episodes_to_plot = []
        
        # If equal sampling mode is active, plot ALL episodes
        if CALIBRATE_ONLY_NEAR_AVS:
            # Plot all episodes when using equal sampling
            for j, (ep, params, _) in enumerate(calibrated, 1):
                episodes_to_plot.append((j, ep, params))
        else:
            # Otherwise, plot subset (first, every 500th, and last)
            for j, (ep, params, _) in enumerate(calibrated, 1):
                if j == 1 or j % 500 == 0 or j == total_episodes:
                    episodes_to_plot.append((j, ep, params))

        print(f"\n[Step 5/7] Creating comparison plots (plotting {len(episodes_to_plot)} out of {total_episodes} episodes)...")
        for plot_idx, (j, ep, params) in enumerate(episodes_to_plot, 1):
            type_label = ep.follower_type.upper() if ep.follower_type != "av" else "AV"
            plot_path = os.path.join(plots_dir, f"episode_{j}_{type_label}_follower_{ep.follower_id}_leader_{ep.leader_id}.png")
            plot_episode_comparison(ep, params, plot_path)
            if plot_idx % 10 == 0 or plot_idx == len(episodes_to_plot):
                print(f" Created {plot_idx}/{len(episodes_to_plot)} plots...")
        print(f" Created plots in: {plots_dir}")

    elif PLOT_COMPARISONS and not MATPLOTLIB_AVAILABLE:
        print("\n[Step 5/7] Skipping plots: matplotlib not available")

    return results

def main():
    # Set random seeds for reproducibility (must be done before any random operations)
    if RANDOM_SEED is not None:
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(RESULTS_DIR, f"calibration_log_{timestamp}.txt")
    tee = Tee(log_file_path)

    try:
        print("=" * 70)
        print("PT Calibration on TGSIM Data (Multiple Datasets)")
        print("=" * 70)
        if RANDOM_SEED is not None:
            print(f" Random seed: {RANDOM_SEED} (results will be reproducible)")
        else:
            print(" WARNING: Random seed not set - results will vary between runs")
        if N_CALIBRATION_RUNS > 1:
            print(f" Calibration mode: {N_CALIBRATION_RUNS} runs per episode ({'best run' if USE_BEST_RUN else 'mean parameters'})")
            print(f" This provides more robust results but takes ~{N_CALIBRATION_RUNS}x longer")
        else:
            print(f" Calibration mode: Single run (fast)")
        print(f"\n Results will be saved to: {RESULTS_DIR}")
        print(f" Console output is being logged to: {log_file_path}")

        print(f"\n Processing {len(CSV_PATHS)} dataset(s):")
        for i, (path, name) in enumerate(zip(CSV_PATHS, DATASET_NAMES), 1):
            print(f" {i}. {name}: {os.path.basename(path)}")

        all_results = []
        for csv_path, dataset_name in zip(CSV_PATHS, DATASET_NAMES):
            dataset_results = process_single_dataset(csv_path, dataset_name)
            all_results.extend(dataset_results)

        if len(all_results) == 0:
            print("\n ERROR: No results from any dataset! Check your data files.")
            return

        print("\n" + "=" * 70)
        print("Combining Results from All Datasets")
        print("=" * 70)

        print("\n Summary by dataset:")
        for dataset_name in DATASET_NAMES:
            dataset_count = sum(1 for r in all_results if r.get("dataset") == dataset_name)
            print(f" - {dataset_name}: {dataset_count} episodes")

        print("\n[Step 6/7] Saving results to CSV and Excel files...")
        res_df = pd.DataFrame(all_results)

        try:
            res_df.to_csv(OUTPUT_EPISODES_CSV, index=False)
            print(f" Wrote episode results: {OUTPUT_EPISODES_CSV}")
            print(f" - {len(res_df):,} episodes with calibrated parameters")
        except PermissionError:
            print(f" WARNING: Cannot write to {OUTPUT_EPISODES_CSV} (file open?)")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(OUTPUT_EPISODES_CSV)[0]
            alt_path = f"{base_name}_{ts}.csv"
            res_df.to_csv(alt_path, index=False)
            print(f" Saved to alternative file: {alt_path}")
        except Exception as e:
            print(f" ERROR: Failed to write CSV file: {e}")

        # Create Excel file with episode summary
        print(f"\n Creating Excel summary file...")
        excel_success = create_episodes_excel(res_df, OUTPUT_EPISODES_EXCEL)
        if excel_success:
            print(f" Created Excel file: {OUTPUT_EPISODES_EXCEL}")
        else:
            # Try alternative filename if permission error
            try:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = os.path.splitext(OUTPUT_EPISODES_EXCEL)[0]
                alt_excel_path = f"{base_name}_{ts}.xlsx"
                if create_episodes_excel(res_df, alt_excel_path):
                    print(f" Created Excel file: {alt_excel_path}")
            except Exception:
                pass

        print("\n[Step 7/7] Generating summary tables...")
        params_table = create_parameters_table_pt(res_df)
        print_formatted_table(params_table, "Table 1: PT Calibrated Parameters by Vehicle Type")

        perf_table = create_performance_table_pt(res_df)
        print_formatted_table(perf_table, "Table 2: PT Performance Metrics by Vehicle Type")

        # -----------------------------
        # PT statistical tests (Welch ANOVA + Games–Howell + Kruskal–Wallis)
        # -----------------------------
        try:
            print("\n[Step 6a/7] Running statistical tests (Welch ANOVA + Games–Howell)...")
            alpha = 0.05

            # --- Welch ANOVA (global 3-group test) ---
            welch_anova_df = run_welch_anova_pt(res_df, alpha=alpha)
            if len(welch_anova_df) > 0:
                print_formatted_table_numeric(
                    welch_anova_df,
                    "Welch ANOVA Results for PT Parameters Across Different Vehicle Types",
                    float_cols=["F-value", "p-value"]
                )
                out_path = os.path.join(RESULTS_DIR, "pt_welch_anova.csv")
                try:
                    welch_anova_df.to_csv(out_path, index=False)
                    print(f" Saved Welch ANOVA to: {out_path}")
                except Exception as e:
                    print(f" WARNING: Could not save Welch ANOVA: {e}")

            # --- Games–Howell post-hoc (pairwise) ---
            gh_df = run_games_howell_pt(res_df, alpha=alpha)
            if len(gh_df) > 0:
                print_formatted_table_numeric(
                    gh_df,
                    "Games–Howell Post-hoc Results for Pairwise Comparisons of PT Parameters",
                    float_cols=["t-Statistic", "df", "p-Value", "Hedges g"]
                )
                out_path = os.path.join(RESULTS_DIR, "pt_games_howell.csv")
                try:
                    gh_df.to_csv(out_path, index=False)
                    print(f" Saved Games–Howell to: {out_path}")
                except Exception as e:
                    print(f" WARNING: Could not save Games–Howell: {e}")

            # --- Optional robustness check: Kruskal–Wallis ---
            kw_df = run_kruskal_pt(res_df, alpha=alpha)
            if len(kw_df) > 0:
                print_formatted_table_numeric(
                    kw_df,
                    "Kruskal–Wallis Robustness Check for PT Parameters Across Vehicle Types",
                    float_cols=["H-stat", "p-value"]
                )
                out_path = os.path.join(RESULTS_DIR, "pt_kruskal_wallis.csv")
                try:
                    kw_df.to_csv(out_path, index=False)
                    print(f" Saved Kruskal–Wallis to: {out_path}")
                except Exception as e:
                    print(f" WARNING: Could not save Kruskal–Wallis: {e}")

        except Exception as e:
            print(f" WARNING: Could not run PT statistical tests: {e}")

        params_table_path = os.path.join(RESULTS_DIR, "pt_parameters_table.csv")
        perf_table_path = os.path.join(RESULTS_DIR, "pt_performance_table.csv")

        try:
            params_table.to_csv(params_table_path, index=False)
            print(f" Saved Table 1 (Parameters) to: {params_table_path}")
        except Exception as e:
            print(f" WARNING: Could not save parameters table: {e}")

        try:
            perf_table.to_csv(perf_table_path, index=False)
            print(f" Saved Table 2 (Performance) to: {perf_table_path}")
        except Exception as e:
            print(f" WARNING: Could not save performance table: {e}")

        summary = (
            res_df[res_df["follower_type"].isin(["small", "large", "av"])]
            .groupby("follower_type")[["Wm", "Alpha", "Beta", "Wc", "Tmax", "Gamma", "fitness", "rmse", "mae", "r_squared"]]
            .agg(["mean", "std", "count"])
        )
        try:
            summary.to_csv(OUTPUT_SUMMARY_CSV)
            print(f" Saved detailed summary to: {OUTPUT_SUMMARY_CSV}")
        except PermissionError:
            print(f" WARNING: Cannot write to {OUTPUT_SUMMARY_CSV} (file open?)")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(OUTPUT_SUMMARY_CSV)[0]
            alt_path = f"{base_name}_{ts}.csv"
            summary.to_csv(alt_path)
            print(f" Saved to alternative file: {alt_path}")
        except Exception as e:
            print(f" ERROR: Failed to write summary CSV file: {e}")

        print("\n" + "=" * 100)
        print("CALIBRATION COMPLETE")
        print("=" * 100)
        print(f"\n Log file saved to: {log_file_path}")

    finally:
        tee.close()
        print(f" Console output has been logged to: {log_file_path}")

if __name__ == "__main__":
    main()
