"""
IDM Calibration on TGSIM using the paper's methodology (Beigi et al.):

- Extract valid car-following episodes:
  * duration > 10 s
  * headway < 200 m
  * no lane change during the episode
  * preceding vehicle ID remains unchanged (leader constant in the episode)
- Calibrate IDM parameters for EACH leader–follower episode via Genetic Algorithm (GA)
- Fitness = sum_j [ w_pos*|x_obs - x_sim| + w_speed*|v_obs - v_sim| ]

Paper basis: "A Data-Driven Comparison of Car-Following Behaviors..." (Beigi et al.)
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

# Results directory: "Results IDM" for equal sampling, "Results Total IDM" for all episodes
RESULTS_DIR = os.path.join(SCRIPT_DIR, "Results IDM" if CALIBRATE_ONLY_NEAR_AVS else "Results Total IDM")

OUTPUT_EPISODES_CSV = os.path.join(RESULTS_DIR, "idm_calib_episodes_results.csv")
OUTPUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "idm_calib_vehicle_type_summary.csv")
OUTPUT_EPISODES_EXCEL = os.path.join(RESULTS_DIR, "idm_calib_episodes_summary.xlsx")
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
# These control the relative importance of position vs speed errors in calibration
# Equal weights (1.0, 1.0) means both are equally important
# Note: Position errors are typically in meters (often 0-10m), speed errors in m/s (often 0-5 m/s)
# If position errors dominate numerically, consider increasing W_SPEED or normalizing errors
# Common choices:
#   - Equal importance: W_POS = 1.0, W_SPEED = 1.0 (current)
#   - Emphasize speed: W_POS = 1.0, W_SPEED = 2.0 or 3.0
#   - Emphasize position: W_POS = 2.0, W_SPEED = 1.0
W_POS = 1.0
W_SPEED = 1

# Episode constraints from paper
MIN_EPISODE_DURATION_S = 10
MAX_HEADWAY_M = 200.0

# Integration settings
DT_MIN = 0.02
DT_MAX = 0.30

# GA settings (optimized for speed; increase if you want tighter calibration)
GA_POP = 50
GA_GENS = 80
GA_ELITE_FRAC = 0.15
GA_TOURN_K = 3
GA_CROSSOVER_PROB = 0.9
GA_MUTATION_PROB = 0.25
GA_MUTATION_SCALE = 0.15  # fraction of parameter range for gaussian mutation
GA_EARLY_STOP_GENS = 10
GA_EARLY_STOP_TOL = 1e-6

# Random seed for reproducibility (set to None for non-deterministic results)
RANDOM_SEED = 42  # Change this value to get different but reproducible results

# Multiple runs configuration for robust calibration
# If > 1: run calibration N times with different seeds and aggregate results
# If 1: single run (faster, but less robust)
N_CALIBRATION_RUNS = 20  # Recommended: 10-30 for robust results, 1 for speed
USE_BEST_RUN = True  # If True: use best run (lowest fitness). If False: use mean of all runs

# Parallel processing configuration
# Number of parallel workers for episode calibration (None = use all CPU cores, 1 = no parallelization)
N_PARALLEL_WORKERS = 12  # Set to None for auto (uses all cores), or specify number (e.g., 4)
# Note: Parallelization gives ~4-8x speedup on multi-core CPUs, but uses more memory
# since each episode calibration is independent and can run in parallel on CPU cores.

# Override from environment (used by run_calibration_sweep.py for W_POS/W_SPEED combos)
if "CALIB_W_POS" in os.environ:
    W_POS = float(os.environ["CALIB_W_POS"])
if "CALIB_W_SPEED" in os.environ:
    W_SPEED = float(os.environ["CALIB_W_SPEED"])
if "CALIB_OUTPUT_SUBFOLDER" in os.environ:
    RESULTS_DIR = os.path.join(SCRIPT_DIR, os.environ["CALIB_OUTPUT_SUBFOLDER"])
    OUTPUT_EPISODES_CSV = os.path.join(RESULTS_DIR, "idm_calib_episodes_results.csv")
    OUTPUT_SUMMARY_CSV = os.path.join(RESULTS_DIR, "idm_calib_vehicle_type_summary.csv")
    OUTPUT_EPISODES_EXCEL = os.path.join(RESULTS_DIR, "idm_calib_episodes_summary.xlsx")
# Quick verification mode (skip heavy calibration)
# - CLI:  python idm_calibration_tgsim_V2.py --stop-after-selection
# - Env:  set CALIB_STOP_AFTER_SELECTION=1
STOP_AFTER_SELECTION = "--stop-after-selection" in sys.argv
if not STOP_AFTER_SELECTION and "CALIB_STOP_AFTER_SELECTION" in os.environ:
    STOP_AFTER_SELECTION = str(os.environ["CALIB_STOP_AFTER_SELECTION"]).strip().lower() not in ("0", "false", "no", "")

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
        cols_lower = {c.lower(): c for c in cols}
        av_strings = ['yes', 'no', 'true', 'false', '1', '0', 'y', 'n']

        def _try_av_column(candidate_col: str, min_ratio: float = 0.5) -> bool:
            """Return True if column looks like an AV flag (Yes/No etc.)."""
            try:
                sample_df = df.sample(min(1000, len(df))) if len(df) > 1000 else df
                if candidate_col not in sample_df.columns:
                    return False
                sample_values = sample_df[candidate_col].dropna().astype(str).str.lower().str.strip()
                if len(sample_values) == 0:
                    return False
                av_count = sample_values.isin(av_strings).sum()
                return (av_count / len(sample_values)) >= min_ratio
            except Exception:
                return False

        # Try "av" and similar first
        av_candidates = ["av", "autonomous", "is_av", "av_flag"]
        for cand in av_candidates:
            if cand in cols_lower:
                candidate_col = cols_lower[cand]
                if _try_av_column(candidate_col, min_ratio=0.5):
                    av_column = candidate_col
                    break

        # Try "acc" (avoid matching acceleration_kf by exact name only)
        if av_column is None and "acc" in cols_lower:
            acc_col = cols_lower["acc"]
            if _try_av_column(acc_col, min_ratio=0.5):
                av_column = acc_col

        # Fallback: for I-90 / I-294, if column "av" or "acc" exists, use it even without validation
        # (TGSIM data often uses these column names for AV flag)
        if av_column is None and dataset_name:
            if "I-90" in dataset_name or "I-94" in dataset_name or "I-294" in dataset_name:
                if "av" in cols_lower:
                    av_column = cols_lower["av"]
                elif "acc" in cols_lower:
                    av_column = cols_lower["acc"]

        # Last resort: guess by name only
        if av_column is None:
            av_column = guess_column(cols, ["av", "autonomous", "is_av", "av_flag"])

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
                    print(f" Longitudinal direction: pos {'increases' if pos_increases_downstream else 'decreases'} downstream (corr d(pos)/dt vs speed = {corr:.3f})")
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
    ascending = [True] * (len(sort_cols) - 1) + [sc.pos_increases_downstream]  # pos ascending iff pos_increases_downstream
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
    # Vehicle ahead: larger pos if pos_increases_downstream, else smaller pos (we sorted so "next" row is ahead)
    ahead = (next_pos > pos_arr) if sc.pos_increases_downstream else (next_pos < pos_arr)
    valid = ~np.isnan(next_veh_id)

    mask = same_group & ahead & valid
    lead_id_arr[mask] = next_veh_id[mask]

    df["lead_id"] = lead_id_arr
    has_leader = (~np.isnan(lead_id_arr)).sum()
    print(f" Completed: {has_leader:,} vehicle-time records have identified leaders ({100*has_leader/len(df):.1f}%)")
    return df

# -----------------------------
# IDM model
# -----------------------------
@dataclass
class IDMParams:
    T: float
    a: float
    b: float
    v0: float
    s0: float
    delta: float

BOUNDS = {
    "T": (0.5, 2.5),
    "a": (0.3, 5.0),
    "b": (0.5, 3.0),
    "v0": (5.0, 35.0),
    "s0": (1.0, 5.0),
    "delta": (3.8, 4.2),
}

# Acceleration hard bounds (match PT_CF_Calibration.ipynb / Talebpour defaults)
ACC_MAX = 5.0
ACC_MIN = -8.0

def idm_acc(v: float, s: float, dv: float, p: IDMParams) -> float:
    v = max(0.0, v)
    s = max(0.1, s)
    sqrt_ab = math.sqrt(max(1e-6, p.a * p.b))
    s_star = p.s0 + max(0.0, v * p.T + (v * dv) / (2.0 * sqrt_ab))
    term_free = (v / max(1e-6, p.v0)) ** p.delta
    term_int = (s_star / s) ** 2
    a_raw = p.a * (1.0 - term_free - term_int)
    return max(ACC_MIN, min(ACC_MAX, a_raw))

def idm_acc_vectorized(v: np.ndarray, s: np.ndarray, dv: np.ndarray, p: IDMParams) -> np.ndarray:
    v = np.maximum(0.0, v)
    s = np.maximum(0.1, s)
    sqrt_ab = np.sqrt(np.maximum(1e-6, p.a * p.b))
    s_star = p.s0 + np.maximum(0.0, v * p.T + (v * dv) / (2.0 * sqrt_ab))
    term_free = (v / np.maximum(1e-6, p.v0)) ** p.delta
    term_int = (s_star / s) ** 2
    a_raw = p.a * (1.0 - term_free - term_int)
    return np.clip(a_raw, ACC_MIN, ACC_MAX)

# NOTE:
# - cache=True can cause failures/corruption when disk is full or when many
#   multiprocessing workers compile/load the same cache concurrently (Windows).
# - We keep JIT on, but disable cache to avoid disk/cache race issues.
@jit(nopython=True, cache=False)
def _simulate_follower_numba(
    dt_arr: np.ndarray,
    v_lead: np.ndarray,
    gap0: np.ndarray,
    x0: float,
    v0: float,
    T: float,
    a: float,
    b: float,
    v0_param: float,
    s0: float,
    delta: float
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(dt_arr) + 1
    x = np.zeros(n)
    v = np.zeros(n)
    x[0] = x0
    v[0] = max(0.0, v0)

    sqrt_ab = np.sqrt(max(1e-6, a * b))
    v0_inv = 1.0 / max(1e-6, v0_param)

    for i in range(n - 1):
        dt = dt_arr[i]
        dv = v[i] - v_lead[i]
        s = gap0[i]
        v_clamped = max(0.0, v[i])
        s_clamped = max(0.1, s)

        s_star = s0 + max(0.0, v_clamped * T + (v_clamped * dv) / (2.0 * sqrt_ab))
        term_free = (v_clamped * v0_inv) ** delta
        term_int = (s_star / s_clamped) ** 2
        a_i = a * (1.0 - term_free - term_int)
        # Hard acceleration bounds (match PT_CF_Calibration.ipynb / Talebpour defaults)
        if a_i > ACC_MAX:
            a_i = ACC_MAX
        elif a_i < ACC_MIN:
            a_i = ACC_MIN

        v_next = max(0.0, v[i] + a_i * dt)
        x_next = x[i] + v_next * dt

        v[i + 1] = v_next
        x[i + 1] = x_next

    return x, v

def simulate_follower(
    t: np.ndarray,
    x_lead: np.ndarray,
    v_lead: np.ndarray,
    x0: float,
    v0: float,
    gap0: np.ndarray,
    p: IDMParams,
) -> Tuple[np.ndarray, np.ndarray]:
    global NUMBA_AVAILABLE
    dt_arr = np.diff(t)
    dt_arr = np.clip(dt_arr, DT_MIN, DT_MAX)

    if NUMBA_AVAILABLE:
        # Be robust: if Numba compilation/import fails in a worker (common with
        # broken SciPy/BLAS installs or corrupted caches), fall back to pure Python.
        try:
            x, v = _simulate_follower_numba(
                dt_arr, v_lead, gap0, x0, v0, p.T, p.a, p.b, p.v0, p.s0, p.delta
            )
            return x, v
        except Exception as e:
            NUMBA_AVAILABLE = False
            # Keep output ASCII-only (Windows console encodings can vary)
            print(f" [WARN] Numba JIT failed ({type(e).__name__}: {e}). Falling back to pure Python simulation.")

    n = len(t)
    x = np.zeros(n, dtype=float)
    v = np.zeros(n, dtype=float)
    x[0] = x0
    v[0] = max(0.0, v0)

    sqrt_ab = np.sqrt(max(1e-6, p.a * p.b))
    v0_inv = 1.0 / max(1e-6, p.v0)

    for i in range(n - 1):
        dt = dt_arr[i]
        dv = v[i] - v_lead[i]
        s = gap0[i]

        v_clamped = max(0.0, v[i])
        s_clamped = max(0.1, s)
        s_star = p.s0 + max(0.0, v_clamped * p.T + (v_clamped * dv) / (2.0 * sqrt_ab))
        term_free = (v_clamped * v0_inv) ** p.delta
        term_int = (s_star / s_clamped) ** 2
        a_i = p.a * (1.0 - term_free - term_int)
        # Hard acceleration bounds (match PT_CF_Calibration.ipynb / Talebpour defaults)
        a_i = max(ACC_MIN, min(ACC_MAX, a_i))

        v_next = max(0.0, v[i] + a_i * dt)
        x_next = x[i] + v_next * dt

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
    # run_index: coerce for grouping, fill NaN with sentinel (do NOT require in dropna)
    if sc.run_index is not None and sc.run_index in df.columns:
        df[sc.run_index] = pd.to_numeric(df[sc.run_index], errors="coerce")
        df[sc.run_index] = df[sc.run_index].fillna(-1).astype("int64")
    # veh_type and av_column: never coerce to numeric here; keep as-is for episode labeling

    # Only require essential trajectory columns (never run_index, veh_type, av_column)
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

    # Avoid pandas FutureWarning: groupby(list_of_len_1) will yield tuple keys in future versions.
    group_keys: str | List[str]
    if "run_index" in merged.columns:
        group_keys = ["run_index", "follower_id"]
    else:
        group_keys = "follower_id"

    for idx_f, (key, g) in enumerate(merged.groupby(group_keys, sort=False), 1):
        g = g.sort_values("t").reset_index(drop=True)
        if isinstance(key, tuple):
            run_i, fid = key
            run_i = int(run_i) if pd.notna(run_i) else None
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
                    # Yes/No-style and column-name-style (acc, av) both mean AV when value indicates yes
                    is_av = av_value in ['yes', 'true', '1', 'y', 'acc', 'av', 'autonomous']
                except Exception:
                    is_av = False

            if is_av:
                follower_type = "av"
            elif isinstance(vt_raw, str) or pd.isna(vt_raw):
                if pd.isna(vt_raw):
                    follower_type = "unknown"
                else:
                    vt_lower = str(vt_raw).lower()
                    if 'small' in vt_lower:
                        follower_type = "small"
                    elif 'large' in vt_lower:
                        follower_type = "large"
                    else:
                        follower_type = "unknown"
            else:
                try:
                    vt = int(vt_raw)
                    follower_type = VEHICLE_TYPE_MAP.get(vt, "unknown")
                except Exception:
                    follower_type = "unknown"

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

def random_params() -> IDMParams:
    return IDMParams(
        T=random.uniform(*BOUNDS["T"]),
        a=random.uniform(*BOUNDS["a"]),
        b=random.uniform(*BOUNDS["b"]),
        v0=random.uniform(*BOUNDS["v0"]),
        s0=random.uniform(*BOUNDS["s0"]),
        delta=random.uniform(*BOUNDS["delta"]),
    )

def params_to_vec(p: IDMParams) -> np.ndarray:
    return np.array([p.T, p.a, p.b, p.v0, p.s0, p.delta], dtype=float)

def vec_to_params(v: np.ndarray) -> IDMParams:
    return IDMParams(T=float(v[0]), a=float(v[1]), b=float(v[2]), v0=float(v[3]), s0=float(v[4]), delta=float(v[5]))

VEC_BOUNDS = np.array([BOUNDS["T"], BOUNDS["a"], BOUNDS["b"], BOUNDS["v0"], BOUNDS["s0"], BOUNDS["delta"]], dtype=float)

def fitness_episode(ep: Episode, p: IDMParams) -> float:
    d = ep.df
    t = d["t"].to_numpy(dtype=float)
    x_lead = d["x_lead"].to_numpy(dtype=float)
    v_lead = d["v_lead"].to_numpy(dtype=float)
    x_obs = d["x_foll"].to_numpy(dtype=float)
    v_obs = d["v_foll"].to_numpy(dtype=float)
    gap = d["gap"].to_numpy(dtype=float)

    x_sim, v_sim = simulate_follower(
        t=t,
        x_lead=x_lead,
        v_lead=v_lead,
        x0=float(x_obs[0]),
        v0=float(v_obs[0]),
        gap0=gap,
        p=p
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

def mutate(v: np.ndarray) -> np.ndarray:
    out = v.copy()
    for i in range(out.size):
        if random.random() < GA_MUTATION_PROB:
            lo, hi = VEC_BOUNDS[i]
            span = hi - lo
            out[i] += random.gauss(0.0, GA_MUTATION_SCALE * span)
            out[i] = clamp(out[i], lo, hi)
    return out

def calibrate_episode_ga(ep: Episode, show_progress: bool = True) -> Tuple[IDMParams, float]:
    pop = [params_to_vec(random_params()) for _ in range(GA_POP)]
    fit = [fitness_episode(ep, vec_to_params(ind)) for ind in pop]
    elite_n = max(1, int(GA_ELITE_FRAC * GA_POP))

    best_fitness_history = []
    no_improvement_count = 0

    for gen in range(GA_GENS):
        order = np.argsort(fit)
        pop = [pop[i] for i in order]
        fit = [fit[i] for i in order]

        best_fitness = fit[0]
        best_fitness_history.append(best_fitness)

        if show_progress and (gen == 0 or gen % 5 == 0 or gen == GA_GENS - 1):
            print(".", end="", flush=True)

        if gen > 0:
            prev_best = best_fitness_history[-2]
            improvement = prev_best - best_fitness
            if improvement < GA_EARLY_STOP_TOL:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            if no_improvement_count >= GA_EARLY_STOP_GENS:
                break

        new_pop = pop[:elite_n]
        while len(new_pop) < GA_POP:
            p1 = tournament_select(pop, fit, GA_TOURN_K)
            p2 = tournament_select(pop, fit, GA_TOURN_K)
            if random.random() < GA_CROSSOVER_PROB:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = p1, p2
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < GA_POP:
                new_pop.append(c2)

        pop = new_pop
        fit = [fitness_episode(ep, vec_to_params(ind)) for ind in pop]

    best_i = int(np.argmin(fit))
    best_p = vec_to_params(pop[best_i])
    best_f = float(fit[best_i])
    return best_p, best_f

def calibrate_episode_robust(ep: Episode, n_runs: int = 1, use_best: bool = True, base_seed: int = None) -> Tuple[IDMParams, float, Dict]:
    """
    Run calibration multiple times and aggregate results.
    
    Args:
        ep: Episode to calibrate
        n_runs: Number of calibration runs
        use_best: If True, return best run (lowest fitness). If False, return mean parameters.
        base_seed: Base seed for runs (each run uses base_seed + run_number)
    
    Returns:
        Tuple of (best_params, best_fitness, stats_dict)
        stats_dict contains: mean_fitness, std_fitness, mean_params, std_params, all_fitnesses
    """
    if n_runs == 1:
        best_p, best_f = calibrate_episode_ga(ep, show_progress=False)
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
        
        params, fitness = calibrate_episode_ga(ep, show_progress=False)
        all_params.append(params)
        all_fitnesses.append(fitness)
    
    all_fitnesses = np.array(all_fitnesses)
    mean_fitness = float(np.mean(all_fitnesses))
    std_fitness = float(np.std(all_fitnesses))
    min_fitness = float(np.min(all_fitnesses))
    max_fitness = float(np.max(all_fitnesses))
    
    # Calculate mean parameters (needed for stats dictionary regardless of use_best)
    mean_T = float(np.mean([p.T for p in all_params]))
    mean_a = float(np.mean([p.a for p in all_params]))
    mean_b = float(np.mean([p.b for p in all_params]))
    mean_v0 = float(np.mean([p.v0 for p in all_params]))
    mean_s0 = float(np.mean([p.s0 for p in all_params]))
    mean_delta = float(np.mean([p.delta for p in all_params]))
    
    if use_best:
        # Return parameters from the run with lowest fitness
        best_idx = int(np.argmin(all_fitnesses))
        best_p = all_params[best_idx]
        best_f = all_fitnesses[best_idx]
    else:
        # Return mean parameters across all runs
        best_p = IDMParams(T=mean_T, a=mean_a, b=mean_b, v0=mean_v0, s0=mean_s0, delta=mean_delta)
        best_f = mean_fitness
    
    # Calculate parameter statistics
    std_T = float(np.std([p.T for p in all_params]))
    std_a = float(np.std([p.a for p in all_params]))
    std_b = float(np.std([p.b for p in all_params]))
    std_v0 = float(np.std([p.v0 for p in all_params]))
    std_s0 = float(np.std([p.s0 for p in all_params]))
    std_delta = float(np.std([p.delta for p in all_params]))
    
    stats = {
        "n_runs": n_runs,
        "mean_fitness": mean_fitness,
        "std_fitness": std_fitness,
        "min_fitness": min_fitness,
        "max_fitness": max_fitness,
        "mean_params": {
            "T": mean_T, "a": mean_a, "b": mean_b, "v0": mean_v0, "s0": mean_s0, "delta": mean_delta
        },
        "std_params": {
            "T": std_T, "a": std_a, "b": std_b, "v0": std_v0, "s0": std_s0, "delta": std_delta
        },
        "all_fitnesses": all_fitnesses.tolist()
    }
    
    return best_p, best_f, stats

# -----------------------------
# Visualization / metrics
# -----------------------------
def calculate_performance_metrics(ep: Episode, params: IDMParams) -> Dict[str, float]:
    d = ep.df
    t = d["t"].to_numpy(dtype=float)
    x_lead = d["x_lead"].to_numpy(dtype=float)
    v_lead = d["v_lead"].to_numpy(dtype=float)
    x_obs = d["x_foll"].to_numpy(dtype=float)
    v_obs = d["v_foll"].to_numpy(dtype=float)
    gap = d["gap"].to_numpy(dtype=float)

    x_sim, v_sim = simulate_follower(
        t=t,
        x_lead=x_lead,
        v_lead=v_lead,
        x0=float(x_obs[0]),
        v0=float(v_obs[0]),
        gap0=gap,
        p=params
    )

    pos_errors = x_obs - x_sim
    rmse = np.sqrt(np.mean(pos_errors ** 2))
    mae = np.mean(np.abs(pos_errors))

    ss_res = np.sum((x_obs - x_sim) ** 2)
    ss_tot = np.sum((x_obs - np.mean(x_obs)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    return {"rmse": float(rmse), "mae": float(mae), "r_squared": float(r_squared)}

def plot_episode_comparison(ep: Episode, params: IDMParams, output_path: str):
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

    x_sim, v_sim = simulate_follower(
        t=t,
        x_lead=x_lead,
        v_lead=v_lead,
        x0=float(x_obs[0]),
        v0=float(v_obs[0]),
        gap0=gap,
        p=params
    )
    
    # Calculate simulated gap (center-based positions)
    gap_sim = x_lead - x_sim - lead_length/2 - follower_length/2

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    type_label = ep.follower_type.upper() if ep.follower_type != "av" else "AV"
    fig.suptitle(
        f"Episode: Follower {ep.follower_id} following Leader {ep.leader_id}\n"
        f"Vehicle Type: {type_label}, Duration: {ep.end_t - ep.start_t:.1f}s\n"
        f"IDM Params: T={params.T:.2f}, a={params.a:.2f}, b={params.b:.2f}, "
        f"v0={params.v0:.2f}, s0={params.s0:.2f}, δ={params.delta:.2f}",
        fontsize=11
    )

    axes[0].plot(t, x_obs, label="Observed", linewidth=2)
    axes[0].plot(t, x_sim, "--", label="Simulated (IDM)", linewidth=2)
    axes[0].plot(t, x_lead, ":", label="Leader (Observed)", linewidth=2, alpha=0.9)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Position (m)")
    axes[0].set_title("Longitudinal Position")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, v_obs, label="Observed", linewidth=2)
    axes[1].plot(t, v_sim, "--", label="Simulated (IDM)", linewidth=2)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_title("Speed")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t, gap, label="Observed Gap", linewidth=2)
    axes[2].plot(t, gap_sim, "--", label="Simulated Gap (IDM)", linewidth=2)
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
def create_parameters_table(res_df: pd.DataFrame) -> pd.DataFrame:
    param_summary = (
        res_df[res_df["follower_type"].isin(["small", "large", "av"])]
        .groupby("follower_type")
        .agg({
            "T": "mean", "a": "mean", "b": "mean",
            "v0": "mean", "s0": "mean", "delta": "mean",
            "follower_type": "count"
        })
        .rename(columns={"follower_type": "count"})
    )

    table_data = []
    params_info = [
        ("T", "T (Desired Time Headway)", f"({BOUNDS['T'][0]}, {BOUNDS['T'][1]})"),
        ("a", "a (Maximum Acceleration)", f"({BOUNDS['a'][0]}, {BOUNDS['a'][1]})"),
        ("b", "b (Desired Deceleration)", f"({BOUNDS['b'][0]}, {BOUNDS['b'][1]})"),
        ("v0", "v0 (Desired Speed)", f"({BOUNDS['v0'][0]}, {BOUNDS['v0'][1]})"),
        ("s0", "s0 (Minimum Desired Gap)", f"({BOUNDS['s0'][0]}, {BOUNDS['s0'][1]})"),
        ("delta", "δ (Acceleration Exponent)", f"({BOUNDS['delta'][0]}, {BOUNDS['delta'][1]})"),
    ]

    for param_key, param_name, param_range in params_info:
        row = {
            "Model": "IDM",
            "Parameter": param_name,
            "Range": param_range,
            "Small Vehicles": f"{param_summary.loc['small', param_key]:.3f}" if 'small' in param_summary.index else "N/A",
            "Large Vehicles": f"{param_summary.loc['large', param_key]:.3f}" if 'large' in param_summary.index else "N/A",
            "Autonomous Vehicles": f"{param_summary.loc['av', param_key]:.3f}" if 'av' in param_summary.index else "N/A",
        }
        table_data.append(row)

    row = {
        "Model": "IDM",
        "Parameter": "count",
        "Range": "-",
        "Small Vehicles": f"{int(param_summary.loc['small', 'count'])}" if 'small' in param_summary.index else "0",
        "Large Vehicles": f"{int(param_summary.loc['large', 'count'])}" if 'large' in param_summary.index else "0",
        "Autonomous Vehicles": f"{int(param_summary.loc['av', 'count'])}" if 'av' in param_summary.index else "0",
    }
    table_data.append(row)

    df_table = pd.DataFrame(table_data)
    return df_table[["Model", "Parameter", "Range", "Small Vehicles", "Large Vehicles", "Autonomous Vehicles"]]

def create_performance_table(res_df: pd.DataFrame) -> pd.DataFrame:
    perf_summary = (
        res_df[res_df["follower_type"].isin(["small", "large", "av"])]
        .groupby("follower_type")
        .agg({"rmse": "mean", "mae": "mean", "r_squared": "mean"})
    )

    table_data = []
    for vtype in ["small", "large", "av"]:
        if vtype in perf_summary.index:
            type_label = "Small Vehicles" if vtype == "small" else ("Large Vehicles" if vtype == "large" else "Autonomous Vehicles")
            row = {
                "Model": "IDM",
                "Vehicle Type": type_label,
                "RMSE": f"{perf_summary.loc[vtype, 'rmse']:.3f}",
                "MAE": f"{perf_summary.loc[vtype, 'mae']:.3f}",
                "R-squared": f"{perf_summary.loc[vtype, 'r_squared']:.4f}",
            }
            table_data.append(row)

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
            excel_df = excel_df.sort_values(["Dataset", "Follower ID", "Start Time (s)"])
        
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
# Statistical tests (Welch ANOVA + Games–Howell + Kruskal–Wallis)
# -----------------------------
def _sig_label(p: float, alpha: float = 0.05) -> str:
    return "Significant Difference" if (p is not None and p < alpha) else "No Significant Difference"


def run_welch_anova_idm(res_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Welch ANOVA (3-group global test, unequal variances OK).
    One-way Welch ANOVA across follower types (small, large, av) for IDM params.
    """
    try:
        import pingouin as pg
    except ImportError:
        print("ERROR: pingouin is required for Welch ANOVA. Install via: pip install pingouin")
        return pd.DataFrame()

    params = ["T", "a", "b", "v0", "s0"]
    types = ["small", "large", "av"]

    df = res_df.copy()
    df = df[df["follower_type"].isin(types)]

    rows = []
    for p in params:
        long_df = df[["follower_type", p]].dropna()
        long_df = long_df.rename(columns={"follower_type": "group", p: "value"})
        if long_df["group"].nunique() < 2 or len(long_df) < 3:
            rows.append({
                "Parameter": p,
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
                "Parameter": p,
                "F-value": f_val,
                "p-value": p_val,
                "Significance": _sig_label(p_val, alpha=alpha)
            })
        except Exception as e:
            rows.append({
                "Parameter": p,
                "F-value": np.nan,
                "p-value": np.nan,
                "Significance": f"Error: {e}"
            })

    return pd.DataFrame(rows, columns=["Parameter", "F-value", "p-value", "Significance"])


def run_games_howell_idm(res_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Games–Howell post-hoc (pairwise, unequal variances OK).
    Returns pairwise comparisons with Hedges g effect size.
    """
    try:
        import pingouin as pg
    except ImportError:
        print("ERROR: pingouin is required for Games–Howell. Install via: pip install pingouin")
        return pd.DataFrame()

    params = ["T", "a", "b", "v0", "s0"]
    df = res_df.copy()
    df = df[df["follower_type"].isin(["small", "large", "av"])]

    out = []
    for p in params:
        long_df = df[["follower_type", p]].dropna()
        long_df = long_df.rename(columns={"follower_type": "group", p: "value"})
        if long_df["group"].nunique() < 2 or len(long_df) < 3:
            continue
        try:
            gh = pg.pairwise_gameshowell(dv="value", between="group", data=long_df)
        except Exception:
            continue
        if gh is None or len(gh) == 0:
            continue
        # pingouin returns columns like: A, B, mean(A), mean(B), diff, se, T, df, pval, hedges
        gh = gh.copy()
        gh["Parameter"] = p
        gh["Significance"] = gh["pval"].apply(lambda x: _sig_label(float(x), alpha=alpha))
        # Standardize column names for output
        rename_map = {
            "A": "Group 1",
            "B": "Group 2",
            "T": "t-Statistic",
            "pval": "p-Value",
            "hedges": "Hedges g"
        }
        out_cols = ["Parameter", "Group 1", "Group 2", "t-Statistic", "df", "p-Value", "Hedges g", "Significance"]
        if "hedges" not in gh.columns:
            gh["hedges"] = np.nan
            rename_map["hedges"] = "Hedges g"
        sel = gh[["Parameter", "A", "B", "T", "df", "pval", "hedges", "Significance"]].rename(columns=rename_map)
        out.append(sel)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)


def run_kruskal_idm(res_df: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """
    Optional robustness check:
    Kruskal–Wallis across follower types (small, large, av) for IDM params.
    Nonparametric (rank-based), useful if distributions are skewed/bounded.
    """
    try:
        from scipy import stats
    except ImportError:
        print("ERROR: scipy is required for Kruskal–Wallis. Install via: pip install scipy")
        return pd.DataFrame()

    params = ["T", "a", "b", "v0", "s0"]
    types = ["small", "large", "av"]

    df = res_df.copy()
    df = df[df["follower_type"].isin(types)]

    rows = []
    for p in params:
        groups = []
        ok = True
        for t in types:
            arr = df.loc[df["follower_type"] == t, p].dropna().to_numpy(dtype=float)
            if len(arr) < 2:
                ok = False
                break
            groups.append(arr)

        if not ok:
            rows.append({
                "Parameter": p,
                "H-stat": np.nan,
                "p-value": np.nan,
                "Significance": "Insufficient data"
            })
            continue

        h_stat, p_val = stats.kruskal(*groups)
        p_val = float(p_val)

        rows.append({
            "Parameter": p,
            "H-stat": float(h_stat),
            "p-value": p_val,
            "Significance": _sig_label(p_val, alpha=alpha)
        })

    return pd.DataFrame(rows, columns=["Parameter", "H-stat", "p-value", "Significance"])


def print_formatted_table_numeric(df: pd.DataFrame, title: str, float_cols: List[str] = None):
    """
    Like your print_formatted_table, but formats selected float columns nicely.
    """
    df2 = df.copy()
    float_cols = float_cols or []
    for c in float_cols:
        if c in df2.columns:
            df2[c] = df2[c].apply(lambda x: f"{x:.6g}" if pd.notna(x) else "NaN")
    print_formatted_table(df2, title)

# -----------------------------
# Parallel processing worker function (IDM)
# -----------------------------
def _calibrate_episode_worker_idm(args: Tuple) -> Tuple[int, Dict, IDMParams, Optional[Dict]]:
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
        best_p, best_fit, calib_stats = calibrate_episode_robust(
            ep,
            n_runs=N_CALIBRATION_RUNS,
            use_best=USE_BEST_RUN,
            base_seed=episode_base_seed
        )
    else:
        best_p, best_fit = calibrate_episode_ga(ep, show_progress=False)
        calib_stats = None

    # Calculate metrics
    metrics = calculate_performance_metrics(ep, best_p)

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
        "fitness": float(best_fit),
        "T": best_p.T,
        "a": best_p.a,
        "b": best_p.b,
        "v0": best_p.v0,
        "s0": best_p.s0,
        "delta": best_p.delta,
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
        result_dict["T_std"] = calib_stats["std_params"]["T"]
        result_dict["a_std"] = calib_stats["std_params"]["a"]
        result_dict["b_std"] = calib_stats["std_params"]["b"]
        result_dict["v0_std"] = calib_stats["std_params"]["v0"]
        result_dict["s0_std"] = calib_stats["std_params"]["s0"]
        result_dict["delta_std"] = calib_stats["std_params"]["delta"]

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
            print(f"\n [WARN] NO AV EPISODES FOUND in {dataset_name}!")
            print(" ERROR: Cannot proceed with equal sampling without AVs.")
            return []

        print(f" [OK] All {n_avs} AV episodes will be used for calibration")

        # Equal sampling around AVs (time overlap ±5s and same lane OR within 100m)
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
    if STOP_AFTER_SELECTION:
        print("\n STOP_AFTER_SELECTION enabled; skipping calibration (verification mode).")
        return []
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

    # Determine number of parallel workers
    n_workers = N_PARALLEL_WORKERS
    if n_workers is None:
        n_workers = cpu_count()
    n_workers = max(1, min(n_workers, total_episodes))  # Don't use more workers than episodes
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
        with Pool(processes=n_workers) as pool:
            completed = 0
            episode_results = {}
            for result in pool.imap_unordered(_calibrate_episode_worker_idm, worker_args):
                episode_idx, result_dict, best_p, calib_stats = result
                episode_results[episode_idx] = (result_dict, best_p, calib_stats)
                completed += 1

                # Progress reporting
                if completed == 1 or completed % 10 == 0 or completed == total_episodes:
                    ep = episodes_to_calibrate[episode_idx - 1]
                    type_label = ep.follower_type.upper() if ep.follower_type != "av" else "AV"
                    print(
                        f" [{completed}/{total_episodes}] Completed {type_label} episode: "
                        f"Follower {ep.follower_id} -> Leader {ep.leader_id} "
                        f"(RMSE={result_dict['rmse']:.3f}m, R²={result_dict['r_squared']:.4f})"
                    )

            # Sort results by episode_idx to maintain order
            for i in range(1, total_episodes + 1):
                if i in episode_results:
                    result_dict, best_p, calib_stats = episode_results[i]
                    results.append(result_dict)
                    ep = episodes_to_calibrate[i - 1]
                    calibrated.append((ep, best_p, dataset_name))
    else:
        # Sequential processing (original behavior)
        for i, ep in enumerate(episodes_to_calibrate, 1):
            show_progress = (i == 1) or (i % 500 == 0) or (i == total_episodes)
            if show_progress:
                type_label = ep.follower_type.upper() if ep.follower_type != "av" else "AV"
                if N_CALIBRATION_RUNS > 1:
                    print(
                        f" [{i}/{total_episodes}] Calibrating {type_label} episode: "
                        f"Follower {ep.follower_id} -> Leader {ep.leader_id} ({N_CALIBRATION_RUNS} runs)...",
                        end="",
                        flush=True
                    )
                else:
                    print(
                        f" [{i}/{total_episodes}] Calibrating {type_label} episode: "
                        f"Follower {ep.follower_id} -> Leader {ep.leader_id}...",
                        end="",
                        flush=True
                    )

            if N_CALIBRATION_RUNS > 1:
                episode_base_seed = None
                if base_seed is not None:
                    episode_base_seed = base_seed + (i - 1) * N_CALIBRATION_RUNS
                best_p, best_fit, calib_stats = calibrate_episode_robust(
                    ep,
                    n_runs=N_CALIBRATION_RUNS,
                    use_best=USE_BEST_RUN,
                    base_seed=episode_base_seed
                )
            else:
                if base_seed is not None:
                    random.seed(base_seed + i - 1)
                    np.random.seed(base_seed + i - 1)
                best_p, best_fit = calibrate_episode_ga(ep, show_progress=show_progress)
                calib_stats = None

            metrics = calculate_performance_metrics(ep, best_p)

            if show_progress:
                if N_CALIBRATION_RUNS > 1 and calib_stats:
                    print(
                        f" Done! (RMSE={metrics['rmse']:.3f}m, R²={metrics['r_squared']:.4f}, "
                        f"fitness={best_fit:.2f}±{calib_stats['std_fitness']:.2f} over {N_CALIBRATION_RUNS} runs)"
                    )
                else:
                    print(f" Done! (RMSE={metrics['rmse']:.3f}m, R²={metrics['r_squared']:.4f})")

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
                "T": best_p.T,
                "a": best_p.a,
                "b": best_p.b,
                "v0": best_p.v0,
                "s0": best_p.s0,
                "delta": best_p.delta,
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r_squared": metrics["r_squared"],
            }

            if N_CALIBRATION_RUNS > 1 and calib_stats:
                result_dict["n_calib_runs"] = calib_stats["n_runs"]
                result_dict["fitness_mean"] = calib_stats["mean_fitness"]
                result_dict["fitness_std"] = calib_stats["std_fitness"]
                result_dict["fitness_min"] = calib_stats["min_fitness"]
                result_dict["fitness_max"] = calib_stats["max_fitness"]
                result_dict["T_std"] = calib_stats["std_params"]["T"]
                result_dict["a_std"] = calib_stats["std_params"]["a"]
                result_dict["b_std"] = calib_stats["std_params"]["b"]
                result_dict["v0_std"] = calib_stats["std_params"]["v0"]
                result_dict["s0_std"] = calib_stats["std_params"]["s0"]
                result_dict["delta_std"] = calib_stats["std_params"]["delta"]

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
        print("IDM Calibration on TGSIM Data (Multiple Datasets)")
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

        # ---- Statistical tests ----
        print("\n[Step 6a/7] Running statistical tests (Welch ANOVA + Games–Howell)...")
        alpha = 0.05

        # --- Welch ANOVA (global 3-group test) ---
        welch_anova_df = run_welch_anova_idm(res_df, alpha=alpha)
        if len(welch_anova_df) > 0:
            print_formatted_table_numeric(
                welch_anova_df,
                "Welch ANOVA Results for IDM Parameters Across Different Vehicle Types",
                float_cols=["F-value", "p-value"]
            )
            out_path = os.path.join(RESULTS_DIR, "idm_welch_anova.csv")
            try:
                welch_anova_df.to_csv(out_path, index=False)
                print(f" Saved Welch ANOVA to: {out_path}")
            except Exception as e:
                print(f" WARNING: Could not save Welch ANOVA: {e}")

        # --- Games–Howell post-hoc (pairwise) ---
        gh_df = run_games_howell_idm(res_df, alpha=alpha)
        if len(gh_df) > 0:
            print_formatted_table_numeric(
                gh_df,
                "Games–Howell Post-hoc Results for Pairwise Comparisons of IDM Parameters",
                float_cols=["t-Statistic", "df", "p-Value", "Hedges g"]
            )
            out_path = os.path.join(RESULTS_DIR, "idm_games_howell.csv")
            try:
                gh_df.to_csv(out_path, index=False)
                print(f" Saved Games–Howell to: {out_path}")
            except Exception as e:
                print(f" WARNING: Could not save Games–Howell: {e}")

        # --- Optional robustness check: Kruskal–Wallis ---
        kw_df = run_kruskal_idm(res_df, alpha=alpha)
        if len(kw_df) > 0:
            print_formatted_table_numeric(
                kw_df,
                "Kruskal–Wallis Robustness Check for IDM Parameters Across Vehicle Types",
                float_cols=["H-stat", "p-value"]
            )
            out_path = os.path.join(RESULTS_DIR, "idm_kruskal_wallis.csv")
            try:
                kw_df.to_csv(out_path, index=False)
                print(f" Saved Kruskal–Wallis to: {out_path}")
            except Exception as e:
                print(f" WARNING: Could not save Kruskal–Wallis: {e}")

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
        params_table = create_parameters_table(res_df)
        print_formatted_table(params_table, "Table 1: IDM Calibrated Parameters by Vehicle Type")

        perf_table = create_performance_table(res_df)
        print_formatted_table(perf_table, "Table 2: IDM Performance Metrics by Vehicle Type")

        params_table_path = os.path.join(RESULTS_DIR, "idm_parameters_table.csv")
        perf_table_path = os.path.join(RESULTS_DIR, "idm_performance_table.csv")

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
            .groupby("follower_type")[["T", "a", "b", "v0", "s0", "delta", "fitness", "rmse", "mae", "r_squared"]]
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
