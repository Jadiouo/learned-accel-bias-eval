#!/usr/bin/env bash
# Re-run the SAME scripts on the MH results (read-only) to prove the port reproduces the paper numbers
# and to give a same-method-subset reference (tcn,m1_cnn,m2,m3_lstm,m4,m6) for the partial Vicon set.
# Needs the MH raw data (default path below, as used by experiments/attitude_controls).
set -euo pipefail
cd "$(dirname "$0")/.."
R=$(cd ../.. && pwd); MHDATA="${1:-${EUROC_DATA_DIR:-$R/data/euroc/machine_hall}}"; OUT=./full_2026-10-23/mh_reference_check
M6=tcn,m1_cnn,m2,m3_lstm,m4,m6; M11=tcn,transformer_s,transformer_m,transformer_l,mamba,m1_cnn,m2,m3_lstm,m4,m5,m6
C=(--results-dir $R/euroc_full/results --config $R/euroc_full/configs/full.yaml --data-dir "$MHDATA" --out-dir $OUT --tag MH-reference)
ATE=$R/experiments/attitude_controls/ate_per_run_gated.csv
export CUDA_VISIBLE_DEVICES=""; RUN=(python3)
"${RUN[@]}" t1_leakage.py "${C[@]}"          # -> t1_TABLE_I_paper_definition.csv (expect 3909/343 ... 1391/122)
"${RUN[@]}" t2_closed_loop.py "${C[@]}"      # -> t2_table_unified.csv (expect 2.977 / 0.105 / 10.976 / 11.165 ...)
"${RUN[@]}" t3_variants.py "${C[@]}"        # ~65 min on MH (4 procs). expect Table IV: O_base 3.03 13.5 [7.5,166]; P_est 85.5; P_off 4.3 [3.8,72]
"${RUN[@]}" t3_spearman.py "${C[@]}"        # expect S1 L=10 delta: IABE 0.94, bias RMSE 0.72 ; abs 0.20 / 0.10
"${RUN[@]}" t3_analyze.py "${C[@]}"         # byte-for-byte equal to experiments/closed_loop/table3_{ratio,state,spearman,shapley}.csv, decomp_summary.csv
"${RUN[@]}" rf_agreement.py "${C[@]}" --ate-csv $ATE --methods $M11 --out-name rf_splithalf_agreement_11.csv   # expect A: raw 0.542, oracle 0.573
"${RUN[@]}" rf_agreement.py "${C[@]}" --ate-csv $ATE --methods $M6  --out-name rf_splithalf_agreement.csv      # 6-method subset
"${RUN[@]}" seed_stability.py "${C[@]}" --ate-csv $ATE --methods $M11 --suffix _11   # expect flip-rate mean 0.773
"${RUN[@]}" seed_stability.py "${C[@]}" --ate-csv $ATE --methods $M6
