# 量測各 DL 方法在 V1_01 真實資料上的訓練 step 速度（batch 64，與 cfg 相同；跳過 warmup）。
import os, sys, time, json
code, config, data_dir = sys.argv[1:4]
nstep = int(sys.argv[4]) if len(sys.argv) > 4 else 60
sys.modules["mamba_ssm"] = None
sys.path.insert(0, os.path.join(code, "src"))
import torch, torch.nn as nn
from euroc_full import common, data_euroc, registry, train as T
cfg = common.load_config(config, data_dir_override=data_dir)
ad = data_euroc.get_adapter(cfg)
d = data_euroc.prepare_regime_a("V1_01", cfg, ad)
bs = cfg["train"]["batch_size"]
spe = (len(d.X_train) + bs - 1) // bs
print(f"V1_01 train windows={len(d.X_train)} steps/epoch={spe} device={T.DEVICE}", flush=True)
X = torch.as_tensor(d.X_train[:bs * 8], dtype=torch.float32); Y = torch.as_tensor(d.y_train[:bs * 8], dtype=torch.float32)
out = {}
for m in [m for m in cfg["methods"] if registry.kind_of(m) == "dl"]:
    model = registry.build_dl_model(m, cfg).to(T.DEVICE).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3); crit = nn.MSELoss()
    ts = []
    for i in range(nstep + 10):
        xb = X[(i % 8) * bs:(i % 8 + 1) * bs].to(T.DEVICE); yb = Y[(i % 8) * bs:(i % 8 + 1) * bs].to(T.DEVICE)
        if T.DEVICE.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(); loss = crit(model(xb), yb); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); loss.item()
        if T.DEVICE.type == "cuda": torch.cuda.synchronize()
        if i >= 10: ts.append(time.time() - t0)
    sps = sum(ts) / len(ts)
    mem = torch.cuda.max_memory_allocated() / 2**30 if T.DEVICE.type == "cuda" else 0
    out[m] = {"s_per_step": sps, "s_per_epoch_est": sps * spe, "peak_gb": mem}
    print(f"{m:14s} {sps*1000:8.1f} ms/step  ~{sps*spe:7.1f} s/epoch(train only)  peak {mem:.2f} GB", flush=True)
    if T.DEVICE.type == "cuda": torch.cuda.reset_peak_memory_stats()
json.dump({"steps_per_epoch": spe, "methods": out}, open("bench.json", "w"), indent=1)
