<h1 align="center"> Papar Car-Following Calibration </h1>

<p align="center">
  <a href="#"><img alt="TRR Paper" src="https://img.shields.io/static/v1?label=TRR%20Paper&message=Under%20Review&color=purple&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/pedrambeigi/Papar_Car-Following_Calibration/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue&style=flat-square"></a>
</p>

---

## Paper & Repository (summary)

**Paper.** *A Data-Driven Comparison of Car-Following Behavior Between Autonomous and Human-Driven Vehicles* examines longitudinal car-following using **TGSIM trajectory data**. Two models are calibrated per follower–leader episode: **IDM** and a **Prospect Theory (PT)** formulation. Calibration quality is judged with trajectory errors (RMSE, MAE); parameter differences across **small vehicles, large vehicles, and AVs** are assessed with **Welch’s ANOVA** and **Games–Howell** post hoc tests.

**Repository.** This codebase implements that pipeline end-to-end: episode extraction from trajectory CSVs, **genetic-algorithm** calibration, optional multiprocessing / Numba, exports (CSV / Excel / plots), and the statistical comparisons above. Defaults target **balanced sampling near AV episodes** so classes are comparable.

## What This Repository Does

At a high level, the repository:
- extracts valid leader-follower car-following episodes from TGSIM trajectories,
- calibrates model parameters for each episode using a Genetic Algorithm (GA),
- evaluates fit quality with trajectory-level error metrics,
- aggregates parameter/performance summaries by vehicle type,
- runs statistical tests across vehicle classes.

## Methodology Implemented in Code

Both calibration pipelines (`IDM` and `PT`) follow the same data-processing logic before model-specific calibration.

### 1) Episode extraction and filtering

For each dataset, the scripts:
- infer the trajectory schema automatically (time, IDs, lane, speed, longitudinal position, vehicle type, optional AV flag, optional vehicle length),
- compute leader IDs if not provided,
- merge follower and leader trajectories by time and leader ID,
- compute bumper-to-bumper gap using vehicle lengths,
- split trajectories into valid episodes.

Each valid episode must satisfy:
- duration > 10 seconds,
- gap > 0 and < 200 meters,
- no lane change in the segment,
- constant leader in the segment.

### 2) Vehicle-type handling and sampling strategy

Episodes are labeled as:
- `small`,
- `large`,
- `av`.

By default, calibration uses **equal sampling near AVs**:
- all AV episodes are included,
- small/large episodes are selected if they are near AV episodes (time overlap and lane/spatial proximity),
- sampled counts for small/large are balanced to AV counts.

This behavior can be switched in code to calibrate all episodes.

### 3) Calibration objective

For both models, the GA minimizes a **weighted sum of absolute position and speed errors** over all time steps *j* in the episode:

`sum_over_j ( W_POS * |x_obs[j] - x_sim[j]| + W_SPEED * |v_obs[j] - v_sim[j]| )`

Weights are set in code as `W_POS` and `W_SPEED` (and can be swept via `run_calibration_sweep.py`).

### 4) Robust calibration mode

Each episode can be calibrated multiple times with different seeds (`N_CALIBRATION_RUNS`), then:
- either keep the best run (`USE_BEST_RUN=True`),
- or use mean parameters across runs.

This improves stability against GA randomness.

### 5) Evaluation and outputs

After calibration, scripts compute:
- RMSE,
- MAE,
- R-squared.

They also export:
- per-episode calibrated parameters and metrics (CSV),
- episode summary (Excel),
- model parameter/performance tables (CSV),
- statistical-test outputs (Welch ANOVA, Games-Howell, Kruskal-Wallis),
- optional observed-vs-simulated comparison plots.

## Data

The analysis uses **Third Generation Simulation (TGSIM)** <a href="https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim?from_hint=eyJxIjoidGdzaW0ifQ%3D%3D"><img alt="Data — TGSIM" src="https://img.shields.io/static/v1?label=Data&message=TGSIM&color=blue&style=flat-square" style="vertical-align: middle;"></a> <a href="https://journals.sagepub.com/doi/10.1177/03611981241257257"><img alt="Paper — TGSIM" src="https://img.shields.io/static/v1?label=Paper&message=TGSIM%20TRR&color=purple&style=flat-square" style="vertical-align: middle;"></a>

TGSIM is public trajectory data from FHWA’s Third Generation Simulation project (e.g., I-395 DC, I-294 IL; see the [Data.gov catalog entry](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim?from_hint=eyJxIjoidGdzaW0ifQ%3D%3D)). Place the trajectory CSVs expected by the scripts under `Dataset/`. The [TGSIM TRR paper](https://journals.sagepub.com/doi/10.1177/03611981241257257) describes the data collection and context.

## Main Scripts

### `idm_calibration_tgsim_V2.py`

Calibrates the **Intelligent Driver Model (IDM)** per episode.

- Calibrated parameters: `T, a, b, v0, s0, delta`.
- Uses GA-based optimization with bounded parameter ranges.
- Simulates follower trajectories with IDM dynamics and acceleration bounds.
- Supports multiprocessing and optional Numba acceleration.
- Produces episode-level and vehicle-type summary files in `Results IDM` (or overridden output subfolder).

### `pt_calibration_tgsim_V2.py`

Calibrates the **Prospect Theory (PT)** car-following model per episode.

- Calibrated parameters: `Wm, Alpha, Beta, Wc, Tmax, Gamma`.
- Uses a Newton-Raphson inner solve for PT acceleration during simulation.
- Uses deterministic simulation in GA fitness for stable optimization.
- Supports multiprocessing and exports the same style outputs as IDM into `Results PT` (or overridden output subfolder).

### `run_calibration_sweep.py`

Runs calibration experiments over multiple weight combinations.

- Sweeps `(W_POS, W_SPEED)` combinations.
- Can run `idm`, `pt`, or `both`.
- Passes settings via environment variables to each calibration script.
- Stores each run in separate subfolders (for clean comparison across objective-weight settings).

## Quick Start

1. Place TGSIM trajectory CSV files in the `Dataset` directory expected by the scripts.
2. Run one model:

```bash
python idm_calibration_tgsim_V2.py
python pt_calibration_tgsim_V2.py
```

3. Run sweep experiments:

```bash
python run_calibration_sweep.py --script both --combos "1,0" "1,1"
```

## Citation

If you use this repository in your research, please consider citing our paper:

```bibtex
@article{beigi2026data,
  title={{A Data-Driven Comparison of Car-Following Behavior Between Autonomous and Human-Driven Vehicles}},
  author={Beigi, Pedram and Rashidi, Mohammad Emad and Li, Nachuan and Bafandkar, Shayan and Monzer, Dana and Hourdos, John and Mahmassani, Hani and Talebpour, Alireza and Hamdar, Samer H.},
  journal={Transportation Research Board - Under Review},
  pages={},
  year={},
  publisher={}
}
```
