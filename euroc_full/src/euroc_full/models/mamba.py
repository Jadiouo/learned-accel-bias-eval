# -*- coding: utf-8 -*-
"""
Mamba — selective state-space（SDD §4，~227K）。

★ `mamba-ssm` 的 CUDA 依賴常爆 → **純 PyTorch fallback 必備**（§3/§4）。
建構時先試 `mamba_ssm.Mamba`（需 CUDA + 編譯）；失敗自動切純 PyTorch 選擇性掃描，
並把 `used_fallback=True` 暴露給 registry/run 記 log。

慣例：input (B, W, 6) → output (B, 3)。純 PyTorch 路徑用順序 scan，causal by construction。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PurePyTorchSSMBlock(nn.Module):
    """精簡版 selective SSM（S6）block，純 PyTorch、causal 順序 scan。"""
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv,
                                groups=self.d_inner, padding=d_conv - 1)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1)     # → dt, B, C
        self.dt_proj = nn.Linear(1, self.d_inner)
        A = torch.arange(1, d_state + 1, dtype=torch.float).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))                    # (d_inner, d_state)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def _scan_readout(self, dt, A, Bm, Cm, xx, chunk=4):
        """selective scan + 讀出，**chunked 平行**（causal、數學等價於逐步遞迴）。

        關鍵：**不materialize整段 (B,W,d_inner,d_state)**——每個 chunk 內用 cumprod/cumsum
        向量化算出 H_chunk，隨即與 C 收縮成 y_chunk (B,c,d_inner) 後丟棄 H_chunk，
        記憶體量級與原逐步版相當；同時把 W=200 步 Python 迴圈降到 W/chunk 次 → ~chunk× 少 kernel。

        chunk 內前綴積 Pbar 的除法在 backward 的梯度 ∝ 1/Pbar²，需 `clamp(dt*A, min=-8)` 保護
        （dA≥exp(-8)≈3.4e-4；只影響「衰減快到幾乎無記憶」的死通道：h_t≈dBx_t 與 dA 值無關，
        故對輸出無實質影響，但讓 forward/backward 在 fp32 下穩定）。chunk=4 數值穩定。
        dt:(B,W,di)  A:(di,ds)  Bm/Cm:(B,W,ds)  xx:(B,W,di) → y:(B,W,di)
        """
        B_, W, di = dt.shape
        ds = A.shape[1]
        h = dt.new_zeros(B_, di, ds)
        ys = []
        for s in range(0, W, chunk):
            sl = slice(s, s + chunk)
            dts = dt[:, sl]                                                  # (B,c,di)
            a = torch.exp(torch.clamp(dts.unsqueeze(-1) * A, min=-8.0))      # (B,c,di,ds)
            b = dts.unsqueeze(-1) * Bm[:, sl].unsqueeze(2) * xx[:, sl].unsqueeze(-1)
            Pbar = torch.cumprod(a, dim=1)                                   # chunk 內前綴積
            Hc = Pbar * (torch.cumsum(b / Pbar, dim=1) + h.unsqueeze(1))     # (B,c,di,ds)
            h = Hc[:, -1]                                                    # 帶到下一個 chunk
            ys.append((Hc * Cm[:, sl].unsqueeze(2)).sum(-1) + self.D * xx[:, sl])
        return torch.cat(ys, dim=1)                                         # (B,W,di)

    def forward(self, x):                       # x: (B, W, d_model)
        B_, W, _ = x.shape
        xz = self.in_proj(x)                    # (B, W, 2*d_inner)
        xx, z = xz.chunk(2, dim=-1)
        xx = self.conv1d(xx.transpose(1, 2))[:, :, :W].transpose(1, 2)   # causal conv
        xx = F.silu(xx)
        proj = self.x_proj(xx)
        dt, Bm, Cm = proj[..., :1], proj[..., 1:1 + self.d_state], proj[..., 1 + self.d_state:]
        dt = F.softplus(self.dt_proj(dt))       # (B, W, d_inner)
        A = -torch.exp(self.A_log)              # (d_inner, d_state)
        y = self._scan_readout(dt, A, Bm, Cm, xx, chunk=4)   # (B, W, d_inner)，因果
        y = y * F.silu(z)
        return self.out_proj(y)


class MambaModel(nn.Module):
    def __init__(self, in_ch=6, d_model=96, n_layers=4, out_dim=3, dropout=0.3):
        super().__init__()
        self.used_fallback = True
        self.in_proj = nn.Linear(in_ch, d_model)
        Block = self._resolve_block()
        self.blocks = nn.ModuleList([Block(d_model) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, out_dim))

    def _resolve_block(self):
        """試官方 mamba-ssm；失敗 → 純 PyTorch fallback（記 used_fallback）。"""
        try:
            import torch as _t
            if not _t.cuda.is_available():
                raise RuntimeError("mamba-ssm 需要 CUDA")
            from mamba_ssm import Mamba                 # noqa: F401
            self.used_fallback = False

            def _wrap(d_model):
                return Mamba(d_model=d_model)
            return _wrap
        except Exception:
            self.used_fallback = True
            return _PurePyTorchSSMBlock

    def forward(self, x):                       # x: (B, W, 6)
        z = self.in_proj(x)
        for blk, norm in zip(self.blocks, self.norms):
            z = z + blk(norm(z))                # pre-norm residual
        return self.head(z[:, -1, :])           # 因果：取最後一步
