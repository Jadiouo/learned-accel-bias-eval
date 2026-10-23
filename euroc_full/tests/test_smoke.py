# -*- coding: utf-8 -*-
"""
Smoke test（SDD §7.1）—— 各方法能 forward/backward；torch 不在場時自動 skip DL 部分。

baseline / PF / linear_stats 為純 numpy，總是測。
"""
import numpy as np
import pytest

from euroc_full import baselines, registry

try:
    import torch as _torch  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

W = 200
CFG = {"window": W, "model": {"dropout": 0.3, "cnn_filters": 64, "hidden_dim": 128,
                              "num_layers": 2, "tcn_channels": [128, 128, 128, 128],
                              "tcn_kernel": 7, "mamba_d_model": 96, "mamba_layers": 4}}
DL = [m for m in registry.list_methods() if registry.is_dl(m)]


# ── 純 numpy baselines（無 torch 也測）──────────────────────────────────────
def test_baselines_run():
    n = 50
    y_train = np.random.randn(200, 3) * 0.01 + 0.02
    assert baselines.constant_pooled(y_train, n).shape == (n, 3)
    assert baselines.constant_perseq(y_train, n).shape == (n, 3)
    kf = baselines.kalman_filter_bias(np.zeros((n, 3)), y_train.mean(0))
    pf = baselines.particle_filter_bias(np.zeros((n, 3)), y_train.mean(0), seed=0)
    assert kf.shape == (n, 3) and pf.shape == (n, 3)
    # PF≈KF（同退化觀測 → 都收斂到 b_init；不 oversell）
    assert np.allclose(kf[-1], pf[-1], atol=5e-2)


def test_linear_stats_fit_predict():
    X = np.random.randn(300, W, 6).astype(np.float32)
    y = X.mean(1)[:, :3] * 0.1 + 0.02
    reg = baselines.LinearStatsRegressor().fit(X, y)
    p = reg.predict(X[:10])
    assert p.shape == (10, 3) and reg.n_params > 0


# ── DL forward/backward（需 torch）──────────────────────────────────────────
@pytest.mark.skipif(not HAVE_TORCH, reason="torch 不在場")
@pytest.mark.parametrize("method", DL)
def test_dl_forward_backward(method):
    import torch
    model = registry.build_dl_model(method, CFG)
    x = torch.randn(4, W, 6); y = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 3)
    loss = ((out - y) ** 2).mean(); loss.backward()
    assert torch.isfinite(loss)
    assert registry.count_params(model) > 0


@pytest.mark.skipif(not HAVE_TORCH, reason="torch 不在場")
@pytest.mark.parametrize("method", ["transformer_s", "transformer_m", "transformer_l"])
def test_transformer_causal_mask(method):
    """擾動 input 後半段，前半段 per-position 預測必不變（§7.1）。"""
    import torch
    model = registry.build_dl_model(method, CFG).eval()
    with torch.no_grad():
        x = torch.randn(2, 40, 6)
        a = model.causal_sequence_output(x)
        x2 = x.clone(); x2[:, 20:] = torch.randn(2, 20, 6)
        b = model.causal_sequence_output(x2)
        assert torch.allclose(a[:, :20], b[:, :20], atol=1e-5)


@pytest.mark.skipif(not HAVE_TORCH, reason="torch 不在場")
def test_mamba_fallback_flag():
    """Mamba 必建得起來；無 CUDA/mamba-ssm 時用純 PyTorch fallback 並記 flag。"""
    model = registry.build_dl_model("mamba", CFG)
    assert hasattr(model, "used_fallback")
    import torch
    out = model(torch.randn(2, W, 6))
    assert out.shape == (2, 3)
