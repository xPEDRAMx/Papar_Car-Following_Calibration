<h1 align="center"> A Data-Driven Comparison of Car-Following Behavior Between Autonomous and Human-Driven Vehicles </h1>

<p align="center">
  <a href="#"><img alt="TRR Paper" src="https://img.shields.io/static/v1?label=TRR%20Paper&message=Under%20Review&color=purple&style=flat-square"></a>&nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://github.com/pedrambeigi/Papar_Car-Following_Calibration/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue&style=flat-square"></a>
</p>

---

## Paper & Repository

**Paper.** *A Data-Driven Comparison of Car-Following Behavior Between Autonomous and Human-Driven Vehicles* examines longitudinal car-following using **TGSIM trajectory data**. Two models are calibrated per follower–leader episode: **IDM** and a **Prospect Theory (PT)** formulation. Calibration quality is judged with trajectory errors (RMSE, MAE); parameter differences across **small vehicles, large vehicles, and AVs** are assessed with **Welch’s ANOVA** and **Games–Howell** post hoc tests.

**Repository.** This codebase implements that pipeline end-to-end: episode extraction from trajectory CSVs, **genetic-algorithm** calibration, optional multiprocessing / Numba, exports (CSV / Excel / plots), and the statistical comparisons above. Defaults target **balanced sampling near AV episodes** so classes are comparable.

## What This Repository Does

At a high level, the repository:

- extracts valid leader–follower car-following episodes from TGSIM trajectories,
- calibrates **Intelligent Driver Model (IDM)**<sup><a href="#note-repo-idm">1</a></sup> and **Prospect Theory (PT)**<sup><a href="#note-repo-pt">2</a></sup> parameters for each episode using a Genetic Algorithm (GA),
- evaluates fit quality with trajectory-level error metrics (RMSE, MAE, R-squared),
- aggregates parameter / performance summaries by vehicle type,
- runs statistical tests<sup><a href="#note-repo-stats">3</a></sup> across vehicle classes.

**Notes:**

<a id="note-repo-idm"></a>**[1] IDM** — parameters: `T, a, b, v0, s0, delta`.

<a id="note-repo-pt"></a>**[2] PT** — parameters: `Wm, Alpha, Beta, Wc, Tmax, Gamma`.

<a id="note-repo-stats"></a>**[3] Statistical tests** — include Welch ANOVA, Games–Howell post hoc tests, and Kruskal–Wallis.

## Data

The analysis uses **Third Generation Simulation (TGSIM)** <a href="https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim?from_hint=eyJxIjoidGdzaW0ifQ%3D%3D"><img alt="Data — TGSIM" src="https://img.shields.io/static/v1?label=Data&message=TGSIM&color=blue&style=flat-square" style="vertical-align: middle;"></a> <a href="https://journals.sagepub.com/doi/10.1177/03611981241257257"><img alt="Paper — TGSIM" src="https://img.shields.io/static/v1?label=Paper&message=TGSIM%20TRR&color=purple&style=flat-square" style="vertical-align: middle;"></a>

TGSIM is public trajectory data from FHWA’s Third Generation Simulation project (e.g., I-395 DC, I-294 IL; see the [Data.gov catalog entry](https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim?from_hint=eyJxIjoidGdzaW0ifQ%3D%3D)). Place the trajectory CSVs expected by the scripts under `Dataset/`. The [TGSIM TRR paper](https://journals.sagepub.com/doi/10.1177/03611981241257257) describes the data collection and context.

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
