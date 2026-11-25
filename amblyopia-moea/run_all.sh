#!/bin/bash
# Full pipeline. Usage: bash run_all.sh <root> [n_jobs]
# <root> must contain raw/<STUDY>/extracted/ (see README.md). All outputs are written below <root>.
# Set QUICK=1 for a reduced-size run that checks the installation end to end (results are not meaningful).
set -euo pipefail
ROOT=${1:?usage: bash run_all.sh <root> [n_jobs]}; NJ=${2:-32}
SRC=$(cd "$(dirname "$0")/src" && pwd)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TSTAR=win12

if [ "${QUICK:-0}" = "1" ]; then
  MCMC="150 150 2"; OPT="2000 2 1000 5"; SENS="2000 2 1000 5"; NCHK=2000
else
  MCMC="2000 2000 4"; OPT="60000 30 20000 200"; SENS="60000 10 20000 100"; NCHK=20000
fi

# 1-2. data harmonisation
python "$SRC/harmonize_aux.py" "$ROOT"
python "$SRC/build_trajectories.py" "$ROOT"

# 3. trajectory model (RQ1): full data + leave-one-trial-out folds, then main-trials-only fits for the sensitivity analysis
for f in ALL ATS5 ATS13 ATS15 ATS16; do python "$SRC/fit_trajectory_model.py" "$ROOT" $f $MCMC; done
python "$SRC/fit_trajectory_model.py" "$ROOT" SUMMARY
for f in ALL ATS5 ATS13 ATS15 ATS16; do NOAUX=1 python "$SRC/fit_trajectory_model.py" "$ROOT" $f $MCMC; done

# 4. static schedules (RQ2, RQ3: NSGA-II, NSGA-III, SMS-EMOA, random search; treatment strata)
python "$SRC/optimize_static.py" "$ROOT" "$NJ" $OPT
python "$SRC/check_simulator.py" "$ROOT" $NCHK

# 5. six further MOEAs (RQ3)
python "$SRC/compare_algorithms.py" "$ROOT" "$NJ" $OPT

# 6. adaptive schedules (RQ4)
python "$SRC/optimize_adaptive.py" "$ROOT" "$NJ" $OPT

# 7. sensitivity analyses and model-free replay
python "$SRC/sensitivity_and_replay.py" "$ROOT" "$NJ" $SENS
