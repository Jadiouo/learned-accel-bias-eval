# analysis_vicon: EuRoC Vicon (V1_01-V2_03) analysis with the same metrics as the MH paper analysis

Status: this README was written while the pipeline was first built on a **partial** Vicon result set (536 ok / 684 runs); the
sections below that mention `partial_2026-10-20` or partial coverage are historical. The final analysis (684/684 runs, K=11 methods)
is `full_2026-10-23/`, and `run_all.sh` now defaults to it. Whole pipeline except `t3_variants.py` takes about 1 minute on CPU.

## Re-run

```bash
cd euroc_full/analysis_vicon
./run_all.sh                                   # default: ../results_vicon -> ./full_2026-10-23 (tag FULL)
./run_all.sh <RESULTS_DIR> <OUT_DIR> <TAG> [DATA_DIR]
full_2026-10-23/run_mh_reference_full.sh       # same scripts on the MH results (port check)
```
Arguments of every script: `--results-dir` (runs_index.csv, runs/*/metrics.json, series.npz), `--data-dir` (raw EuRoC folders,
default `../kaggle_vicon/data_vicon`), `--config` (default `configs/vicon.yaml`), `--out-dir`, `--tag`, `--min-seeds` (default 10).
Every CSV starts with a `# [TAG] results_dir=...` line (read with `pd.read_csv(..., comment="#")`).
t1 peak RSS about 0.3 GB (measured). The scripts were originally run under a memory cap; the cap wrapper was removed from the shell scripts for the public release.

**Method set rule.** Cross-method statistics (R-F, seed stability) use the deep methods that have `>= --min-seeds` ok seeds on every
sequence (complete-case). Today that is `tcn, m1_cnn, m2, m3_lstm, m4, m6` (K=6, 15 pairs). With the full set it becomes all 11 (55 pairs)
automatically. `coverage.md` in the output folder lists exactly what is missing.

## Where the MH numbers come from, and what was ported

| Paper (taes_2026-10) | MH source (read-only) | Port here | Change |
|---|---|---|---|
| Table I (leakage by attitude source: theta_bar, leakage/IABE, g*theta/eps), IV-A | `experiments/attitude_controls/run_controls_gated.py` (outputs `table_sources_regime_summary_gated.csv`, `ate_per_run_gated.csv`) | `t1_leakage.py` -> `t1_TABLE_I_paper_definition.csv` (+ the full `t1_*` set) | Copied, then: paths/sequences from config, results loaded via `vcommon.load_wide`, dose-response plot dropped, only regime A exists, extra block applies the paper's Table I definition (median leakage / pooled regime IABE median). Reproduces `gyro_ate` to 1e-15 on V (script's own check). |
| Table II (closed-loop budget: b_ref C_abs, est C_abs, est C_Delta, zero C_Delta, filter-self row), IV-B | `experiments/closed_loop/run_closed_loop.py` (mode O, cfg base) + `unify_table.py` -> `table_unified.csv`; `eskf.py` | `t2_closed_loop.py` -> `t2_table_unified.csv`; `eskf.py` is a byte-identical copy | Copied; only mode O / cfg base; 5 noise reps as MH; sequences, results path from config. |
| Table III row O_base (shared-error / between-estimator ratio, CI) | `experiments/closed_loop/analyze_3.py` (`units`, `fr`, `boot`) | `t2_closed_loop.py` -> `t2_table3_ratio_obase.csv` | Copied functions; sequence-cluster bootstrap over 6 sequences (MH 5). |
| IV-C-1 / Fig. 4 direction agreement (R-F, split-half, ATE vs bias RMSE ranking) | `paper/revision_2026-10/verification/R-F_noncircular/rF_noncircular_stratified.py` | `rf_agreement.py` -> `rf_splithalf_agreement.csv` | Copied; K methods instead of 11; groups = ALL + each sequence (no A/B split). |
| IV-C-3 seed instability (0.77) and Fig. 5 | `.../R-B_scripts/tost_backfill.py` (flip part) | `seed_stability.py` -> `seed_flip_rate.csv` | Same formula `1 - sum c(c-1)/(n(n-1))`. Also reports it for bias RMSE (Fig. 5 ranks by bias RMSE; the 0.77 is on IABE). |
| IV-C-2 split-half reliability; IV-C-4 resolution (213/495, MDE, seeds needed) | `.../R-E_stats/rE_stats.py` (parts a, b, c) | `seed_stability.py` -> `seed_reliability.csv`, `seed_resolution_by_seq.csv`, `seed_mde.csv`, `seed_pairwise_iabe.csv`, `seed_rho_vs_iabe.csv` | Copied formulas (paired t + BH within the unit's pairs, d>=2 rule, MDE by noncentral t at 80% power). "Seeds needed" uses Bonferroni over the K(K-1)/2 pairs instead of 0.05/55. |
| (support) metrics_wide.csv | `euroc_full/src/euroc_full/analysis.py` `load_runs/write_long_wide` | `vcommon.py` / `build_wide.py` | Same logic (ok runs only, metric pivot). |

### Port verification (`partial_2026-10-20/mh_reference_check/`, made by `run_mh_reference.sh`)
The same scripts run on the MH results reproduce the paper values: Table I leakage multiples 3909/343, 75/6.5, 344/30, 1391/122, 11.6/1.0
and g*theta/eps 5506/724 ...; Table II 0.635/2.977/10.976 (b_ref C_abs), 0.105 and 0.895 (est C_Delta), 1.90/7.58/31.0 (zero); Table III O_base L=10 common 3.03 m, ratio 13.5 [7.5, 166]; R-F regime A raw 0.542, oracle 0.573, Mahony 0.557, gated 0.545, static 0.418, IABE 0.822, ceiling 0.988; flip rate mean 0.773 (0.64-0.87); 122/275 distinguishable pairs in regime A.
The same folder also holds the MH numbers on the 6-method subset that is complete in the partial Vicon data (suffix without `_11`), for a like-for-like reference.

## Ported in the second batch (2026-10-20)
| Paper | MH source (read-only) | Port | MH check (`mh_reference_check/`) |
|---|---|---|---|
| Table III rows P_est, P_off, O_q10, O_q01; Table IV; onset-state r_tilt/r_vel | `run_pre_outage.py` (task_P, task_V), `analyze_3.py` | `t3_variants.py` -> `t3_pre_outage_long.csv`, `t3_variants_long.csv`; `t3_analyze.py` -> `t3_table3_ratio.csv`, `t3_table3_state.csv` | equals `experiments/closed_loop/table3_ratio.csv`, `table3_state.csv` (max abs diff 6e-16 / 0); Table IV values 3.03, 13.5 [7.5,166], 85.5, 4.29 ... |
| Shapley shares (Fig. 3), base ungated/gated | `decompose.py`, `summarize_decomp.py`, task_V part of `run_pre_outage.py` | `t3_variants.py` (`t3_decomp_units.csv`, `t3_variants_decomp_units.csv`), `t3_analyze.py` -> `t3_table3_shapley.csv`, `t3_decomp_summary.csv` | equals `table3_shapley.csv`, `decomp_summary.csv` (max diff 9e-16) |
| Spearman IABE / bias RMSE vs C_Delta, C_abs (S1) | `run_closed_loop.py build_predictors`, `analyze.py` | `t3_spearman.py` -> `t3_predictors.csv`, `t3_spearman_scenario.csv`; within-sequence Spearman in `t3_table3_spearman.csv` | L=10 delta 0.943 / 0.721, abs 0.204 / 0.100 (paper 0.94 / 0.71-0.72, 0.20 / 0.10); within-seq equals `table3_spearman.csv` |
Changes vs the originals: `rc`/`rg` replaced by the t2 port and a verbatim copy of `run_gated.outage_prop`; ATE-based predictors dropped in `t3_spearman.py`; its bootstrap RNG is re-seeded per scenario, so S1 CIs (not point values) differ from `scenario_stats.csv`. The CIs in t3_analyze are the original (single RNG, same call order) and match to 1e-15. Cost: `t3_variants.py` 64 min on MH, 25 min on partial V (4 procs, peak RAM < 1 GB).

## Not ported (no Vicon equivalent yet)
- Onset-tilt tables beyond `t3_table3_state.csv`, gated-consequence tables (`run_gated.py` per-source, `analyze_gated.py`, S1 gated variants: 0.92/0.76 in the paper), D-mode (injection at i0) rows.
- Regime B (pooled training) and all A/B-stratified or 9-unit pooled statistics: `configs/vicon.yaml` has regime A only. So the "nine cells" Spearman becomes 6 cells (V, regime A).
- TUM-VI items, supplementary tables, `make_figs.py` figures: no V counterpart built.
- No figures are generated (CSV only).

## V1_01 accelerometer-bias anomaly (evidence summary)
GT `b_a` of V1_01 is 0.548 m/s^2 in y (constant, std 0.002). Pipeline `b_ref` equals the GT file (mean [-0.014 0.548 0.068]). At rest (|v|<0.03, |w-bw|<0.03, GT attitude) the accelerometer minus R^T g is [-0.021 0.509 0.055] (first 2 s: 0.472), so the number is consistent with the IMU data; it is not a unit/axis/alignment bug. At L=20 onset the ESKF has tilt error -2.7 deg (g*theta_y = -0.47) and b_a,y estimate 0.12: the filter absorbed the offset into tilt, so injecting b_ref double-compensates (C_abs 66.8 m vs zero bias 16.9 m).

## Alignment limits (read before comparing with MH)
1. Only 6 of 11 deep methods are complete, so pair counts (15 vs 55), the random-winner baseline (1-1/K) and agreement rates are not comparable to the 11-method MH values. Use the MH 6-method reference in `mh_reference_check/`.
2. V has no regime B. The paper's A/B split, the pooled 9-unit 0.56 (p=0.13) and B-only claims have no V analogue. V has 6 sequences in 2 rooms (V1 x3, V2 x3); for sequence-cluster bootstraps they are treated as 6 clusters. Rooms are not modeled.
3. Test spans are shorter (V 12.4-21.4 s, MH 14.7-27.2 s) and IABE grows with T^2, so absolute IABE (m) and the pooled-IABE-based leakage multiples are not comparable across datasets.
4. Closed loop: the aided pre-outage segment is about 73-123 s on V (V1_02 about 73 s), against "100 s or more" on MH. L=20 s exists for 1 sequence on V (V1_01) vs 2 on MH; treat L=20 as descriptive.
5. "learned+const" pools all ok runs; the partial set is unbalanced (mamba 39, transformer_m 20, transformer_s 38, m5 55, transformer_l 0), n=536 vs 1026 on MH.
6. b_ref is the dataset's EKF bias state in both datasets, not a direct measurement. V1_01 has a b_ref accelerometer-bias y component of 0.55 m/s^2, about 4x the others (0.10-0.14).
7. KF and PF equal the constant baseline by design (no label input), as in MH.
