#!/usr/bin/env bash
# Re-run the whole Vicon analysis. CPU only, ~1 minute. Usage:
#   ./run_all.sh [RESULTS_DIR] [OUT_DIR] [TAG] [DATA_DIR]
# Partial run (today):   ./run_all.sh                       (defaults below)
# Full run (later):      ./run_all.sh ../results_vicon_full  ./full_<date>  FULL
set -euo pipefail
cd "$(dirname "$0")"
RES="${1:-../results_vicon}"; OUT="${2:-./full_2026-10-23}"; TAG="${3:-FULL}"; DATA="${4:-../kaggle_vicon/data_vicon}"
ARGS=(--results-dir "$RES" --out-dir "$OUT" --tag "$TAG" --data-dir "$DATA")
export CUDA_VISIBLE_DEVICES=""
RUN=(python3)
"${RUN[@]}" build_wide.py     "${ARGS[@]}"   # metrics_wide.csv + coverage
"${RUN[@]}" t1_leakage.py     "${ARGS[@]}"   # Table I (leakage by attitude source) + ATE per run
"${RUN[@]}" t2_closed_loop.py "${ARGS[@]}"   # Table II (closed-loop budget) + Table III O_base ratio row
"${RUN[@]}" t3_variants.py    "${ARGS[@]}"   # modes P_est/P_off, Q x10/x0.1, Shapley units (slowest: ~1 h on the partial set, 4 procs)
"${RUN[@]}" t3_spearman.py    "${ARGS[@]}"   # IABE / bias RMSE vs C_Delta, C_abs Spearman (+ predictors)
"${RUN[@]}" t3_analyze.py     "${ARGS[@]}"   # Table III variants / Table IV, onset state, Shapley shares (Fig. 3), within-seq Spearman
"${RUN[@]}" rf_agreement.py   "${ARGS[@]}"   # R-F direction agreement (split-half)
"${RUN[@]}" seed_stability.py "${ARGS[@]}"   # flip rate, resolution, reliability
echo "done -> $OUT"
