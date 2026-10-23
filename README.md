# learned-accel-bias-eval

Code and summary results for the paper

> **MEMS Accelerometer Bias, Tilt, and the Position-Error Budget of Learned Bias Estimation**
> Hao-Ting Ho. Submitted to IEEE Transactions on Aerospace and Electronic Systems (under review).

The paper shows how a shared attitude (tilt) error leaks gravity into the accelerometer-bias evaluation, how large
this leakage is for common attitude sources, and how much of the closed-loop position error after a GNSS outage is
shared by all bias estimators rather than caused by the estimator choice. The repository exists so that reviewers and
readers can re-derive the numbers in the paper.

## What is and is not in this repository

Included: all source code, configs, the Kaggle kernel scripts used for training, the analysis scripts, and the small
summary CSV files that the analysis scripts and the figure/number generators read (about 16 MB in total).

Not included (too large, or third-party data):

| Item | Why | How to obtain |
|---|---|---|
| EuRoC MAV raw data | third-party dataset | see "Getting EuRoC" |
| `euroc_full/results*/runs/` (per-run `metrics.json`, `series.npz`, model outputs) | several hundred MB | re-train, see "Reproducing" |
| Model weights, logs, caches | not needed for the analysis | - |

Summary files that **are** included (so that table/figure scripts run without re-training):

- `euroc_full/results/runs_index.csv`, `euroc_full/results/metrics_wide.csv`: index and per-run metrics of the 684 EuRoC Machine Hall (MH) runs (11 deep methods x 10 seeds x regimes, plus baselines).
- `euroc_full/results_vicon/runs_index.csv`: index of the 684 Vicon-room (V1/V2) runs.
- `euroc_full/analysis_vicon/full_2026-10-23/`: all output CSVs of the V1/V2 analysis (final, 684/684 runs) and the MH port-check (`mh_reference_check/`).
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
  attitude_controls/     Table I: attitude sources (raw, oracle, static average, Mahony, gated Mahony); vqf/ (VQF sources)
  closed_loop/           Tables II-IV: ESKF outage experiment, Shapley decomposition, CIs; inekf/ (InEKF robustness)
  sim_truth/             known-truth simulation
verification/            statistics scripts (R-B flip rate, R-E reliability/resolution, R-F direction agreement)
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

## Where the paper's numbers come from

| Paper item | Script | Output file(s) | Needs |
|---|---|---|---|
| Training, evaluation, IABE/ATE metrics | `euroc_full/src/euroc_full/{run,train,metrics,data_euroc}.py`, `configs/full.yaml` | `euroc_full/results/{runs_index,metrics_wide}.csv` | raw data + GPU |
| Table I and Fig. 2/dose response (MH): leakage by attitude source, theta, g*theta/eps | `experiments/attitude_controls/run_controls_gated.py` (`run_controls.py` = ungated version) | `table_sources_regime_summary_gated.csv`, `dose_points_gated.csv`, `ate_per_run_gated.csv` | raw data + `series.npz` |
| Table I VQF rows, supplementary VQF table | `experiments/attitude_controls/vqf/run_vqf.py`, `vqf/v2/run_v2.py` | `vqf/v2/table_sources_*_v2.csv`, `agreement_v2.csv` | raw data + `series.npz`, `vqf` |
| Table II: closed-loop budget (C_abs, C_Delta) | `experiments/closed_loop/run_closed_loop.py`, `unify_table.py`, `eskf.py` | `consequences_long.csv`, `table_unified.csv` | raw data + `series.npz` |
| Table III/IV: shared-error / between-estimator ratio, variants P_est, P_off, O_q10, O_q01 | `run_pre_outage.py`, `analyze_3.py`, `ratio_ci.py`, `ci_seedmean.py` | `table3_ratio.csv`, `table3_state.csv`, `table_ratio_ci.csv`, `ci_seedmean.csv` | raw data + `series.npz` |
| Shapley shares (Fig. 3), Spearman S1 | `decompose.py`, `summarize_decomp.py`, `analyze.py` | `table3_shapley.csv`, `decomp_summary.csv`, `table3_spearman.csv`, `scenario_stats.csv` | raw data |
| Gated-tilt variants (supplementary) | `run_gated.py`, `analyze_gated.py`, `sens_gated.py` | `table_gated*.csv`, `gated_consequences.csv` | raw data + `series.npz` |
| InEKF robustness | `experiments/closed_loop/inekf/{harness,inekf,analysis,verdict}.py`, `test_inekf.py` | `inekf/out_inekf/*.csv` | raw data |
| Fig. 4 direction agreement (split-half), fig. data | `verification/R-F_noncircular/rF_noncircular_stratified.py` | `part2_splithalf_agreement_stratified.csv` | summary CSVs only |
| Seed instability (flip rate), Fig. 5 | `verification/R-B_scripts/tost_backfill.py` | `flip_rate.csv` | summary CSVs only |
| Split-half reliability, resolution (MDE, seeds needed) | `verification/R-E_stats/rE_stats.py` | `verification/R-E_stats/*.csv` | summary CSVs only |
| Learned vs. constant baseline | `verification/learned_vs_const.py` | `learned_vs_const.csv` | summary CSVs only |
| Known-truth simulation | `experiments/sim_truth/{sim_truth,run_all,analyze}.py` | `results_tables.md`, `results_summary.json` | MH ground truth |
| V1/V2 (Vicon) results: Tables I-IV analogues, agreement, seed statistics | `euroc_full/analysis_vicon/run_all.sh` (`t1_leakage`, `t2_closed_loop`, `t3_variants`, `t3_spearman`, `t3_analyze`, `rf_agreement`, `seed_stability`) | `euroc_full/analysis_vicon/full_2026-10-23/*.csv` | raw data + `series.npz` |
| V1/V2 number macros in the paper | `paper_scripts/make_vicon_numbers.py` | LaTeX macros (stdout) | `full_2026-10-23/` CSVs (included) |
| Figures and supplementary tables | `paper_scripts/make_figs.py` | `paper_scripts/figs/*.pdf`, `supp_table_*.tex` | summary CSVs (included) |
| Port check (V pipeline on MH reproduces the MH numbers) | `euroc_full/analysis_vicon/full_2026-10-23/run_mh_reference_full.sh` | `full_2026-10-23/mh_reference_check/` | raw data + `series.npz` |

## Notes

- Comments and some docstrings in the source are in Chinese (the working language of the project); identifiers and outputs are English.
  References such as "SDD section X" point to an internal design document that is not part of this repository.
- `verification/R-B_scripts/tost_backfill.py` also looks for TUM-VI results (`euroc_full/results/tumvi`), which are not distributed here; the EuRoC part
  works from the included files. `configs/tumvi.yaml` is kept for completeness.
- `make_figs.py` regenerates the supplementary tables `supp_table_leak.tex` and `supp_table_flip.tex` with caption text that differs slightly from the
  hand-edited versions in the manuscript; the numeric content of the four supplementary tables comes from the same CSV files.
- Machine-specific absolute paths were replaced by paths relative to the repository root and the `EUROC_DATA_DIR` environment variable. Header
  comments in some CSV/JSON outputs show `<REPO>` where the original absolute path was.

## License

MIT, see `LICENSE`.
