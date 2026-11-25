<h1 align="center">amblyopia-moea</h1>

<p align="center">
  <b>Simulation-based multi-objective optimization of visit intervals and plateau-declaration rules in amblyopia treatment</b>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10-blue">
  <img alt="NumPyro" src="https://img.shields.io/badge/NumPyro-0.15.3-orange">
  <img alt="pymoo" src="https://img.shields.io/badge/pymoo-0.6.1.5-green">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows%20(WSL)-lightgrey">
</p>

<p align="center">
  <a href="#overview">Overview</a> &bull;
  <a href="#installation">Installation</a> &bull;
  <a href="#data">Data</a> &bull;
  <a href="#quick-start">Quick start</a> &bull;
  <a href="#pipeline">Pipeline</a> &bull;
  <a href="#schedule-encoding">Schedule encoding</a> &bull;
  <a href="#reproducibility">Reproducibility</a>
</p>

---

This repository contains the complete analysis code accompanying our manuscript.
A single command runs every analysis reported in the paper, from the raw public trial data to the summary tables.

## Overview

Children treated for amblyopia are re-examined at fixed intervals until visual acuity (VA) stops improving.
This project treats follow-up as a three-objective problem and searches visit timing and the stopping rule jointly:

| Objective | Definition |
|---|---|
| `f1` visits | mean number of follow-up visits per child |
| `f2` delay | mean time from the true plateau `t*` to the declared plateau, `max(0, stop - t*)` |
| `f3` premature stop | proportion of children in whom a plateau is declared before `t*` |

The pipeline

1. **harmonizes** public Pediatric Eye Disease Investigator Group (PEDIG) trial data into long-format VA trajectories;
2. **fits** a Bayesian nonlinear mixed-effects exponential-plateau model of amblyopic-eye VA (NumPyro, NUTS), validated leave-one-trial-out;
3. **simulates** virtual cohorts with heavy-tailed measurement noise rounded to 0.1 logMAR, as on a line-based chart;
4. **optimizes** schedules with multi-objective evolutionary algorithms (pymoo), selects one schedule by an epsilon-constraint rule, and compares it with the PEDIG trial rule on cohorts calibrated to held-out trials.

### Features

- **Vectorized evaluator.** All candidate schedules are scored on the same simulated children with common random numbers; an independent child-by-child implementation is included as a check.
- **Nine MOEAs and a random-search baseline under one protocol.** NSGA-II, NSGA-III, SMS-EMOA, SPEA2, RVEA, AGE-MOEA, AGE-MOEA-II, IBEA and SPEA2+SDE all use the same encoding, operators, budget and external archive.
- **Leave-one-trial-out design.** Schedules are always selected on training trials and evaluated on a held-out trial.
- **Adaptive schedules.** Intervals that depend on the gain just observed can be compared with static schedules.
- **Sensitivity analyses.** Measurement noise, plateau threshold, horizon and model specification are varied, and a model-free replay on observed trajectories is included.
- **Resumable runs.** Every optimization run is checkpointed, so an interrupted step continues where it stopped.

## Repository structure

```
amblyopia-moea/
├── README.md
├── requirements.txt
├── run_all.sh                     # full pipeline (QUICK=1 for a reduced test run)
└── src/
    ├── harmonize_aux.py           # step 1: auxiliary trials -> derived/aux_*.csv
    ├── build_trajectories.py      # step 2: trajectories, patients, same-day retests
    ├── fit_trajectory_model.py    # step 3: Bayesian trajectory model (RQ1)
    ├── optimize_static.py         # step 4: static schedules (RQ2, RQ3); also the simulator module
    ├── check_simulator.py         # step 4: independent check of the evaluator
    ├── compare_algorithms.py      # step 5: six further MOEAs (RQ3)
    ├── optimize_adaptive.py       # step 6: adaptive schedules (RQ4)
    └── sensitivity_and_replay.py  # step 7: sensitivity analyses and model-free replay
```

## Installation

Python 3.10 is required. A virtual environment is recommended.

```bash
git clone https://github.com/FionaRuan89/amblyopia-moea.git
cd amblyopia-moea
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

All dependencies are pinned in `requirements.txt` (NumPy, SciPy, pandas, JAX, NumPyro, pymoo). JAX runs on the CPU; no GPU is needed.

## Data

The trial data are **not redistributed** here. Download the public datasets from the Jaeb Center for Health Research
(<https://public.jaeb.org/pedig/stdy>) and unzip each study into

```
<root>/raw/<STUDY>/extracted/
```

using the study folder names below. The scripts find the data tables inside `extracted/` by file name.

| Folder | Role in the analysis |
|---|---|
| `ATS05`, `ATS13`, `ATS15`, `ATS16` | main trials (model fitting, held-out evaluation) |
| `ATS1`, `ATS2A`, `ATS2B`, `ATS3`, `ATS06`, `ATS10` | auxiliary trials (model fitting only) |
| `ATS12` | weekly visits, used for the model-free replay |

> **Attribution.** Publications based on these data must include the study-specific attribution given in each study's README and the statement: "The analyses content and conclusions presented herein are solely the responsibility of the authors and have not been reviewed or approved by PEDIG."

## Quick start

Check the installation with a reduced-size run of the whole pipeline (about 10 minutes with 32 workers; the numbers are not meaningful):

```bash
QUICK=1 bash run_all.sh /path/to/root 32
```

Reproduce the full analysis:

```bash
bash run_all.sh /path/to/root 32
```

The second argument is the number of worker processes (default 32). All outputs are written below `<root>`. The full run is computationally heavy, because each optimization step performs several hundred runs of 60,000 evaluations, so a multi-core machine is recommended.

## Pipeline

| Step | Script | Output (below `<root>`) | Manuscript |
|---|---|---|---|
| 1 | `harmonize_aux.py` | `derived/aux_*.csv` | Table 1 (auxiliary trials) |
| 2 | `build_trajectories.py` | `derived/traj.csv`, `pts.csv`, `retest.csv` | Section 2.1, Table 1 |
| 3 | `fit_trajectory_model.py` | `results/trajectory_model/` | Section 3.1, Table 3 (RQ1) |
| 4 | `optimize_static.py` | `results/static/` | Section 3.2, Table 4 (RQ2); Section 3.3 (RQ3, part) |
| 4 | `check_simulator.py` | printed | Section 2.4 (retest calibration); Section 3.2 (simulator check) |
| 5 | `compare_algorithms.py` | `results/algorithms/` | Section 3.3, Table 5, Table S1 (RQ3) |
| 6 | `optimize_adaptive.py` | `results/adaptive/` | Section 3.4, Table S2 (RQ4) |
| 7 | `sensitivity_and_replay.py` | `results/sensitivity/` | Section 3.5, Tables 6 and 7 |

Each results folder contains a `summary.txt` with the numbers reported in the manuscript.

Every script can also be run on its own; its usage line is given in the module docstring, for example

```bash
python src/fit_trajectory_model.py <root> <fold> [warmup samples chains]
python src/optimize_static.py <root> <n_jobs> <n_eval> <n_seeds> [n_train] [n_rep]
```

`optimize_static.py` also serves as a module. The simulator (`cohort`), the vectorized evaluator of a schedule (`evaluate`), the epsilon-constraint selection rule (`select`) and the held-out evaluation (`test_eval`) are imported by the later steps.

### Environment variables

| Variable | Values | Effect |
|---|---|---|
| `TSTAR` | `win12` (default), `inf` | plateau definition: expected gain over the next 12 weeks below one line, or remaining total gain below one line |
| `NOAUX` | `1` | `fit_trajectory_model.py` fits the main trials only (sensitivity analysis) |
| `RUNS_ONLY` | `1` | `compare_algorithms.py` stops after the optimization runs, without the summary |
| `QUICK` | `1` | `run_all.sh` runs the reduced-size test configuration |

## Schedule encoding

A static schedule is the integer vector

```
(d1, d2, d3, d4, d5, d6, L, m, Tmin)
```

| Gene | Range | Meaning |
|---|---|---|
| `d1`..`d6` | 2-16 weeks | the first six intervals between visits; `d6` is repeated until the 52-week horizon |
| `L` | 1-2 lines | improvement threshold, in lines of 0.1 logMAR |
| `m` | 1-3 | number of consecutive non-improving visits that declares a plateau |
| `Tmin` | 0-16 weeks | minimum treatment time before a plateau can be declared |

The code `m = 0` denotes the rule of the PEDIG trials, the comparator `(5, 5, 5, 5, 5, 5, 1, 0, 0)`, which lies outside the search space. Visits take place every 5 weeks, and each acuity is compared with the previous visit. When acuity has not improved by at least one line, it is retested the same day and the better value is kept. A plateau is declared at the first visit that still shows no one-line improvement. The retest error has correlation `RHO_RT = 0.25` with the first measurement (`optimize_static.py`).

The two-consecutive-visit rule with fixed intervals, e.g. `(5, 5, 5, 5, 5, 5, 1, 2, 0)`, is evaluated as a secondary reference.

An adaptive schedule (`optimize_adaptive.py`) uses seven integers `(d1, A, B, C, L, m, Tmin)`. `A`, `B` and `C` are the next interval after a visit whose acuity improved by at least two lines, by exactly one line, or not at all.

## Reproducibility

- All random seeds are fixed in the scripts, and every optimization run has its own seed.
- Package versions are pinned in `requirements.txt`.
- `run_all.sh` sets `OMP_NUM_THREADS=1` (and the OpenBLAS and MKL equivalents), so each worker process uses a single thread.
- Each optimization run is saved under `results/*/runs/` as soon as it finishes. If a step is interrupted, run it again: completed runs are loaded from these files, and only the missing runs are computed.
