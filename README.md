# learned-accel-bias-eval

Code and summary results for the paper

> **Evaluating Learned Accelerometer-Bias Estimators by Trajectory Error: Tilt Leakage and a Closed-Loop Error Budget**
> Hao-Ting Ho. Manuscript in preparation (not yet submitted).

The paper asks whether open-loop trajectory error can tell which learned accelerometer-bias estimator is more
accurate. On EuRoC, it compares the ranking by trajectory error with the ranking by bias error against the dataset's
reference bias, measures how much gravity a tilt error leaks into the position error for several attitude sources, and
estimates how much of the closed-loop position error after a simulated outage is shared by all bias estimators rather
than caused by the estimator choice. The repository exists so that reviewers and
readers can re-derive the numbers in the paper. Table and figure numbers below follow the manuscript version of 2026-11-26
(Table I: setup; II: definitions; III: gravity leakage; IV: position-error budget; V: robustness;
Fig. 1: error flow; Fig. 2: agreement chain; Fig. 3: heatmap; Fig. 4: scatter; Fig. 5: Shapley shares). "S" numbers refer to the supplementary
material of the same version.

**Recommendation of the paper.** Where a reference bias exists, report the bias error as two numbers, the offset (the mean error over the
evaluation window) and the fluctuation around it, each with paired seed intervals. Rank estimators by the offset when the aim is navigation, and
give the leakage ratio (the leakage acceleration g*theta, with theta the mean tilt error of the attitude source, in multiples of the rms bias error) next to every trajectory-error result. Open-loop trajectory error
should be reported with its attitude source and leakage ratio and should not be used alone to rank estimators; the benefit for navigation needs a closed-loop
test. The evidence behind this: on EuRoC Machine Hall the rankings by trajectory error and by bias error agreed on 0.56 of the separable pairs (95% interval 0.46 to 0.66),
removing the shared leakage term raised the agreement to 0.80, and the reference bias was nearly constant in the test windows, so the bias error is mostly an offset
(supplement S17-S19). The evidence is partly post hoc and covers EuRoC Machine Hall, forward recursive filters and an estimator-output reference bias.

## What is and is not in this repository

Included: all source code, configs, the Kaggle kernel scripts used for training, the analysis scripts, and the small
summary CSV files that the analysis scripts and the figure/number generators read (about 56 MB in total).

Not included (too large, or third-party data):

| Item | Why | How to obtain |
|---|---|---|
| EuRoC MAV raw data | third-party dataset | see "Getting EuRoC" |
| `euroc_full/results*/runs/` (per-run `metrics.json`, `series.npz`, model outputs) | several hundred MB | re-train, see "Reproducing" |
| Model weights, logs, caches | not needed for the analysis | - |

Summary files that **are** included (so that table/figure scripts run without re-training):

- `euroc_full/results/runs_index.csv`, `euroc_full/results/metrics_wide.csv`: index and per-run metrics of the 684 EuRoC Machine Hall (MH) runs (11 deep methods x 10 seeds x regimes, plus baselines).
- `euroc_full/results_vicon/runs_index.csv`: index of the 684 Vicon-room (V1/V2) runs.
- `euroc_full/analysis_vicon/full_2026-10-23/`: the output CSVs of the V1/V2 analysis (final, 684/684 runs), including the per-run ATE table `t1_ate_per_run_gated.csv` (1.4 MB), and the MH port-check (`mh_reference_check/`).
- `experiments/reviewer_checks/`: scripts and result CSV/JSON of the additional checks in supplementary S6, S10 and S13-S19 and the canonical agreement table H0 (see the layout below; about 22 MB, the largest are `F6_ba_overwrite/src_long.csv` 5 MB, `F3_dose_response/dose_const_long.csv` 3 MB and `F5_scaled_attitude/ate_scaled_per_run.csv` 3 MB).
- `experiments/attitude_controls/*.csv`, `experiments/closed_loop/*.csv`, `experiments/closed_loop/inekf/**.csv`, `experiments/attitude_controls/vqf/**.csv`: outputs of the MH attitude-source and closed-loop experiments (the largest are `consequences_long.csv` 6.9 MB, `gated_consequences.csv` 2.2 MB, `ate_per_run*.csv` 1.7-2.1 MB).
- `verification/`: split-half agreement, seed-stability and resolution statistics (scripts and CSVs).

## Layout

```
euroc_full/
  src/euroc_full/        package: data (EuRoC adapter), models (TCN, Transformer S/M/L, Mamba, CNN/LSTM variants),
                         train.py, metrics.py (IABE, ATE), run.py (orchestrator), analysis.py
  configs/               full.yaml (MH, regimes A/B, 10 seeds), vicon.yaml (V1/V2), smoke.yaml, tumvi.yaml
  tests/                 metric sanity tests and smoke tests
  kaggle_vicon/          Kaggle kernel scripts: train/ (worker, bench, main), train_kernel*/ (kernel entry points for the
                         training stages and the resume stages), download/ (EuRoC V1/V2 IMU+GT downloader)
  analysis_vicon/        V1/V2 analysis pipeline (t1..t3, rf_agreement, seed_stability), run_all.sh, final outputs
experiments/
  attitude_controls/     Table III: attitude sources (raw, oracle, static average, Mahony, gated Mahony); vqf/ (VQF sources)
  closed_loop/           Tables IV-V: ESKF outage experiment, Shapley decomposition, CIs; inekf/ (InEKF robustness)
  sim_truth/             known-truth simulation
  reviewer_checks/       additional checks, one folder per supplementary section, each with a REPRODUCE.txt:
                         F4_seqcluster (S6, sequence-cluster intervals), F7_vicon_decomp (S10 and S16.1, Vicon decomposition),
                         F1_noise_ceiling (S13), F2_aligned_ate (S14), F3_dose_response (S15), decomposition (S16.1),
                         F5_scaled_attitude (S16.2), F6_ba_overwrite (S16.3), F8_what_is_learned (S17),
                         F9_openloop_dose (S18), F10_offset_reference (S19),
                         H0_unified_stats (canonical agreement table: one pipeline for every agreement rate printed in the paper)
verification/            statistics scripts (R-B flip rate, R-E reliability/resolution, R-F direction agreement, mamba_exclusion[_vicon])
paper_scripts/           make_figs.py (figures and supplementary tables), make_vicon_numbers.py (LaTeX number macros)
```

## Getting EuRoC

Download the EuRoC MAV dataset (Burri et al., IJRR 2016) from ETH: <https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets>
(also on the ETH Research Collection). Only `mav0/imu0/data.csv` and `mav0/state_groundtruth_estimate0/data.csv` of each sequence are used.

- Machine Hall (MH_01_easy .. MH_05_difficult): place under `data/euroc/machine_hall/` or set `EUROC_DATA_DIR=/path/to/machine_hall`.
- Vicon rooms (V1_01_easy .. V2_03_difficult): place under `euroc_full/kaggle_vicon/data_vicon/` or pass `--data-dir` to the analysis scripts (set `EUROC_DATA_DIR` for training).
- `euroc_full/kaggle_vicon/download/main.py` is the Kaggle CPU kernel that fetched only the IMU and ground-truth files of V1/V2 via HTTP range requests (about 60 MB instead of 12 GB). It writes to `/kaggle/working/vicon_room`.

Expected layout: `<DATA_DIR>/<SEQUENCE>/mav0/{imu0,state_groundtruth_estimate0}/data.csv`.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e euroc_full            # optional; scripts also add euroc_full/src to sys.path
```
Python 3.12 was used (3.9+ should work). The analysis, figure and number scripts are CPU only. `vqf` is needed only for the VQF experiment.

## Hardware

Model training (11 deep methods x 10 seeds x 6 V sequences, and the MH matrix) ran on Kaggle Tesla T4 GPUs (up to 2 x T4 per kernel
session). All analysis scripts run on a CPU; the slowest, `analysis_vicon/t3_variants.py`, takes roughly 25-65 min with 4 processes.

## Reproducing

1. **Training** (GPU, optional): `python -m euroc_full.run --config euroc_full/configs/full.yaml --full` (MH) and
   `... --config euroc_full/configs/vicon.yaml --full` (V1/V2). Runs are resumable. The Kaggle kernels in `euroc_full/kaggle_vicon/`
   are the scripts that were used to run it in stages on Kaggle (private Kaggle datasets holding the code and data are referenced by
   name in the original kernel metadata, which is not published here).
2. **Per-run results needed by the closed-loop scripts** (`series.npz` with the predicted bias per run) come from step 1.
   Everything downstream of the included summary CSVs runs without them (see the last column of the table below).
3. **Verification without re-training** (fast, CPU):
   ```bash
   python paper_scripts/make_vicon_numbers.py euroc_full/analysis_vicon/full_2026-10-23 > vicon_numbers.tex   # all V1/V2 macros
   python paper_scripts/make_figs.py                                                                           # figures + supp. tables
   pytest -q euroc_full/tests/test_metric_sanity.py
   ```
4. **Additional checks** (supplementary S6, S10, S13-S19 and H0): see `REPRODUCE.txt` in each folder of `experiments/reviewer_checks/`. The aggregation
   and summary scripts run from the included CSV files; the scripts that rebuild trajectories or rerun the filter need EuRoC raw data and `series.npz`.

## Where the paper's numbers come from

| Paper item | Script | Output file(s) | Needs |
|---|---|---|---|
| Training, evaluation, IABE/ATE metrics | `euroc_full/src/euroc_full/{run,train,metrics,data_euroc}.py`, `configs/full.yaml` | `euroc_full/results/{runs_index,metrics_wide}.csv` | raw data + GPU |
| Table III and supplementary Fig. S1 (dose response), MH: leakage by attitude source, theta, g*theta/eps | `experiments/attitude_controls/run_controls_gated.py` (`run_controls.py` = ungated version) | `table_sources_regime_summary_gated.csv`, `dose_points_gated.csv`, `ate_per_run_gated.csv` | raw data + `series.npz` |
| Table III VQF rows, supplementary VQF tables (S8, Tables S6-S7) | `experiments/attitude_controls/vqf/run_vqf.py`, `vqf/v2/run_v2.py` | `vqf/v2/table_sources_*_v2.csv`, `agreement_v2.csv` | raw data + `series.npz`, `vqf` |
| Table IV and Fig. 4: closed-loop budget (C_abs, C_Delta) | `experiments/closed_loop/run_closed_loop.py`, `unify_table.py`, `eskf.py` | `consequences_long.csv`, `table_unified.csv` | raw data + `series.npz` |
| Table V: shared-error / between-estimator ratio, variants P_est, P_off, O_q10, O_q01 | `run_pre_outage.py`, `analyze_3.py`, `ratio_ci.py`, `ci_seedmean.py` | `table3_ratio.csv`, `table3_state.csv`, `table_ratio_ci.csv`, `ci_seedmean.csv` | raw data + `series.npz` |
| Shapley shares (Fig. 5), rank correlation with C_abs (Table S12) | `decompose.py`, `summarize_decomp.py`, `analyze.py` | `table3_shapley.csv`, `decomp_summary.csv`, `table3_spearman.csv`, `scenario_stats.csv` | raw data |
| Lookahead check: estimate applied from the end of its input window (supplement S11, Table S13) | `experiments/closed_loop/lookahead_check/{run,analyze,analyze_P_full,check_shift}.py` (see `REPRODUCE.txt` there) | `summary_C.csv`, `summary_P_full.csv`, `A_ratio_compare.csv` | `out_*.csv` (included); `run.py` needs raw data + `series.npz` |
| Gated-tilt variants (supplement S4-S5, Tables S2-S3) | `run_gated.py`, `analyze_gated.py`, `sens_gated.py` | `table_gated*.csv`, `gated_consequences.csv` | raw data + `series.npz` |
| InEKF robustness (supplement S9, Table S8) | `experiments/closed_loop/inekf/{harness,inekf,analysis,verdict}.py`, `test_inekf.py` | `inekf/out_inekf/*.csv` | raw data |
| Canonical direction agreement of ATE and IABE with the bias-RMSE ranking (every agreement rate printed in the paper and the supplement: all 252 seed splits, leave-one-sequence-out jackknife intervals with t quantiles; with and without Mamba; seed-noise ceilings) | `experiments/reviewer_checks/H0_unified_stats/01_reproduce.py` ... `06_fig2.py` (see `REPRODUCE.txt` there) | `canonical_agreement.csv`, `fig2_rows.csv`, `ceiling_*.csv`, `differences.csv` | summary CSVs only |
| Split-half direction agreement of ATE and IABE with the bias-RMSE ranking (original pipeline, superseded by the canonical table above for every printed number; `reviewer_checks/F1_noise_ceiling/common.py` and `F2_aligned_ate/03_agreement.py` re-implement it) | `verification/R-F_noncircular/rF_noncircular_stratified.py` | `part2_splithalf_agreement_stratified.csv` | summary CSVs only |
| Fig. 2: agreement chain (bias-RMSE retest, IABE, ATE without the leakage term, raw ATE, random-direction leakage, SE(3)-aligned ATE, noise ceiling) | `paper_scripts/make_figs.py` (`chain_data`, `fig_agree`) | `paper_scripts/figs/fig4_agreement.pdf` | `experiments/reviewer_checks/H0_unified_stats/{canonical_agreement,fig2_rows}.csv` (included) |
| Seed instability (flip rate), Table S5 | `verification/R-B_scripts/tost_backfill.py` | `flip_rate.csv` | summary CSVs only |
| Sequence-cluster bootstrap intervals of the agreement rates (supplement S6) | `experiments/reviewer_checks/F4_seqcluster/{f4,f4_ci}.py` | `f4_results.csv`, `f4_ci.csv` | summary CSVs only |
| Seed-noise ceiling of the agreement (supplement S13, Table S15) | `experiments/reviewer_checks/F1_noise_ceiling/{reproduce,simulate}.py` | `results_ceiling_N2000.csv`, `results_calibration_N2000.csv`, `reproduce_observed.csv` | summary CSVs only |
| Alignment of the open-loop trajectory before ATE (supplement S14, Table S16) | `experiments/reviewer_checks/F2_aligned_ate/01_build_ate.py`, `02_repro.py`, `03_agreement.py`, `04_align_selftest.py` | `ate_aligned_per_run.csv`, `agreement_aligned*.csv`, `ate_magnitudes.csv` | `03` and `04`: included CSVs; `01`: raw data + `series.npz` |
| Closed-loop dose response to an injected bias error (supplement S15, Tables S17-S18) | `experiments/reviewer_checks/F3_dose_response/01_repro.py` ... `05_extras.py` | `dose_*_long.csv`, `thresholds*.csv`, `poolsim_spearman.csv`, `pool_runs_summary.csv` | `03`, `05`: included CSVs; `01`, `02`, `04`: raw data + `series.npz` |
| Decomposition of the position error into three terms (supplement S16.1, Table S19) | `experiments/reviewer_checks/decomposition/{f5_decomp,jk,f15_checks}.py` | `f5_decomp_runs.csv`, `f5_decomp_agree.csv` | `jk.py`: included CSVs; `f5_decomp.py`: raw data + `series.npz` |
| Vicon-room decomposition and Mamba-exclusion agreement (supplement S10, S16.1) | `experiments/reviewer_checks/F7_vicon_decomp/{vicon_runs,vicon_agree,mh_nomamba}.py` | `vicon_decomp_runs.csv`, `vicon_agree.csv`, `mh_nomamba_agree.csv` | `vicon_agree.py`, `mh_nomamba.py`: included CSVs; `vicon_runs.py`: `results_vicon/runs/*/series.npz` + V1/V2 raw data |
| Rescaling of the attitude error (supplement S16.2, Table S20) | `experiments/reviewer_checks/F5_scaled_attitude/{01_build,02_analyze}.py` | `ate_scaled_per_run.csv`, `results_agreement.csv`, `rm_scales.json` | `02`: included CSVs; `01`: raw data + `series.npz` |
| Overwriting the filter's bias state at outage onset (supplement S16.3, Table S21) | `experiments/reviewer_checks/F6_ba_overwrite/01_run.py` ... `04_checks.py` | `cond_long.csv`, `src_long.csv`, `table_*.csv` | `02`, `03`: included CSVs; `01`, `04`: raw data + `series.npz` |
| What the bias error measures in the evaluation windows: offset/fluctuation split, tracking skill, agreement when each estimate is replaced by its window mean (supplement S17, Table S22; post hoc) | `experiments/reviewer_checks/F8_what_is_learned/{extract_stats,analysis_a,analysis_b,analysis_c}.py` | `stats_per_run.csv`, `stats_enriched.csv`, `item1_*.csv`, `item23_by_method.csv`, `item4_pairs.csv`, `item6_agree.csv` | `analysis_*`: included CSVs; `extract_stats.py`: `series.npz` |
| Open-loop dose response: agreement of ATE with the bias-RMSE order for two synthetic constant estimators whose bias RMSE differs by Delta, per attitude source, with the Delta at which it returns to 0.8 and its relation to the tilt leakage (supplement S18, Fig. S3, Table S23; post hoc; VQF with bias clip 10 deg/s, cold start) | `experiments/reviewer_checks/F9_openloop_dose/01_build.py` ... `04_analyze.py`, `f9_lib.py` (see `REPRODUCE.txt` there) | `dose_summary.csv`, `dose_curves_pooled.csv`, `dose_counts.csv`, `leakage_by_source.csv`, `crossing_vs_leakage.csv`, `real_pair_scale.json`, `repro_check.json`, `selftest_offsets.json` | `make_figs.py` reads the included CSV/JSON; `02`-`04` also need the intermediate files `ops.pkl` and `dose_chunks.npz` (not included); `01`, `02b`: raw data + `series.npz`, `vqf` |
| Direction agreement when the reference ranking is the window-mean offset instead of bias RMSE, with and without the shared leakage term (supplement S19, Table S24; post hoc; Machine Hall) | `experiments/reviewer_checks/F10_offset_reference/01_offset_agreement.py` (see `REPRODUCE.txt` there) | `offset_reference_agreement.csv`, `per_unit_agreement.csv` | summary CSVs only (included); about 1 min |
| Split-half reliability, resolution (MDE, seeds needed) | `verification/R-E_stats/rE_stats.py` | `verification/R-E_stats/*.csv` | summary CSVs only |
| Mamba-exclusion sensitivity, MH (supplement S6; item (6) of S12): split-half agreement, Spearman ATE-IABE, flip rate and closed-loop ratio recomputed with the 11 deep methods and with the 10 without Mamba | `verification/mamba_exclusion/mamba_exclusion.py`, `mamba_pair_share.py` | `agreement.csv`, `spearman_ate_iabe.csv`, `flip.csv`, `distinguishable.csv`, `closed_loop.csv`, `mamba_pair_share.csv` | summary CSVs, including `experiments/closed_loop/{pre_outage_long,variants_long}.csv` (produced by `run_pre_outage.py`) |
| Same Mamba-exclusion check on V1/V2 (supplement S6; item (6) of S12) | `verification/mamba_exclusion_vicon/mamba_exclusion_vicon.py` | `agreement.csv`, `agreement_per_seq.csv`, `flip.csv`, `distinguishable.csv`, `closed_loop.csv`, `mamba_outlier_ratio.csv` | per-run `euroc_full/results_vicon/runs/*/metrics.json` (not included, from training), `euroc_full/analysis_vicon/full_2026-10-23/t1_ate_per_run_gated.csv` (included), and `t3_pre_outage_long.csv`, `t3_variants_long.csv` in the same folder (not included; produced by `t3_variants.py`, need `series.npz`) |
| Learned vs. constant baseline | `verification/learned_vs_const.py` | `learned_vs_const.csv` | summary CSVs only |
| Known-truth simulation (supplement S3, Fig. S2) | `experiments/sim_truth/{sim_truth,run_all,analyze}.py` | `results_tables.md`, `results_summary.json` | MH ground truth |
| V1/V2 (Vicon) results: Tables III-V analogues (supplement S10, Tables S9-S11), agreement, seed statistics | `euroc_full/analysis_vicon/run_all.sh` (`t1_leakage`, `t2_closed_loop`, `t3_variants`, `t3_spearman`, `t3_analyze`, `rf_agreement`, `seed_stability`) | `euroc_full/analysis_vicon/full_2026-10-23/*.csv` | raw data + `series.npz` |
| V1/V2 number macros in the paper | `paper_scripts/make_vicon_numbers.py` | LaTeX macros (stdout) | `full_2026-10-23/` CSVs (included) |
| Figures (Figs. 1-5, S1-S3, graphical abstract) and supplementary tables S1-S3, S5, S15-S24 | `paper_scripts/make_figs.py` | `paper_scripts/figs/*.pdf`, `supp_table_*.tex` | summary CSVs (included) |
| Port check (V pipeline on MH reproduces the MH numbers) | `euroc_full/analysis_vicon/full_2026-10-23/run_mh_reference_full.sh` | `full_2026-10-23/mh_reference_check/` | raw data + `series.npz` |

## Notes

- Comments and some docstrings in the source are in Chinese (the working language of the project); identifiers and outputs are English.
  References such as "SDD section X" point to an internal design document that is not part of this repository.
- `verification/R-B_scripts/tost_backfill.py` also looks for TUM-VI results (`euroc_full/results/tumvi`), which are not distributed here; the EuRoC part
  works from the included files. `configs/tumvi.yaml` is kept for completeness.
- `make_figs.py` regenerates the fourteen supplementary tables `supp_table_*.tex` (S1-S3, S5, S15-S24); their captions and numeric content match the
  manuscript version of 2026-11-26 and come from the included CSV files. Every agreement rate it prints is read from `experiments/reviewer_checks/H0_unified_stats/canonical_agreement.csv`.
- `paper_scripts/make_vicon_numbers.py` reads the V1/V2 direction agreements from the same canonical table (it no longer uses `analysis_vicon/full_2026-10-23/rf_splithalf_agreement.csv`, the original pipeline, which is kept only for reference).
- Machine-specific absolute paths were replaced by paths relative to the repository root and the `EUROC_DATA_DIR` environment variable. Header
  comments in some CSV/JSON outputs show `<REPO>` where the original absolute path was.

## License

MIT, see `LICENSE`.
