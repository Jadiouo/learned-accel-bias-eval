# -*- coding: utf-8 -*-
"""EuRoC Full Matrix — Paper 1.A 主結果套件（self-contained，GitHub-ready）。

承接 SDD_EuRoC_MVP_11（MVP 凍結於 `euroc_mvp/`）。本套件對 MVP **零 import 相依**：
metric / data / baselines 核心由 MVP 驗證版**複製**而來，邏輯不改（見 tests/ 的 regression）。

公開介面（pinned）：
  common.load_config(path, data_dir_override) -> cfg
  data_euroc.get_adapter(cfg) / prepare_regime_a(seq,cfg) / prepare_regime_b_pooled(cfg)
  metrics.build_metric_inputs(...) / iabe_rot / iabe_body / gyro_ate / bias_rmse / compute_all_metrics
  baselines.constant_pooled / constant_perseq / linear_stats(LinearStatsRegressor) / kalman_filter_bias / particle_filter_bias
  registry.list_methods() / is_dl(name) / build_dl_model(name,cfg) / count_params
  plan.build_plan(cfg)
  train.train_model(model,data,cfg,seed) / infer(model,X)
  analysis.run_analysis(cfg)
  seeding.seed_everything(seed, torch_deterministic)
"""
__version__ = "1.0.0"
