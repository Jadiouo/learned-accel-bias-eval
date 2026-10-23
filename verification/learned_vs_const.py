#!/usr/bin/env python3
"""Regime A: constant baseline IABE vs learned estimators (canonical IABE = iabe_rot).
Per sequence: const = constant_perseq (single run); per learned method = median over seeds;
best learned = min over the 11 deep methods; learned median = median over the 11 per-method medians.
Re-run: python3 learned_vs_const.py"""
import pandas as pd, os
here = os.path.dirname(os.path.abspath(__file__))
src = os.path.join(here, '..', 'euroc_full', 'results', 'metrics_wide.csv')
d = pd.read_csv(src)
p = d.run_id.str.split('__', expand=True)
d['reg'], d['method'], d['seq'] = p[0], p[1], p[2]
a = d[d.reg == 'A']
deep = ['tcn','transformer_s','transformer_m','transformer_l','mamba','m1_cnn','m2','m3_lstm','m4','m5','m6']
assert set(deep) <= set(a.method)
rows = []
for s in sorted(a.seq.unique()):
    g = a[a.seq == s]
    const = g[g.method == 'constant_perseq'].iabe_rot
    assert len(const) == 1
    c = float(const.iloc[0])
    pm = g[g.method.isin(deep)].groupby('method').iabe_rot.median()
    assert len(pm) == 11
    best = pm.idxmin(); b = float(pm.min()); med = float(pm.median())
    rows.append(dict(seq=s, const_iabe=c, best_method=best, best_learned_iabe=b,
                     learned_median_iabe=med, ratio_const_over_best=c/b,
                     ratio_const_over_learned_median=c/med,
                     n_seeds_per_method=int(g[g.method.isin(deep)].groupby('method').size().min())))
o = pd.DataFrame(rows)
o.to_csv(os.path.join(here, 'learned_vs_const.csv'), index=False)
pd.set_option('display.width', 250); print(o.to_string())
print('ratio best: %.3f-%.3f; ratio median: %.3f-%.3f' % (o.ratio_const_over_best.min(), o.ratio_const_over_best.max(),
      o.ratio_const_over_learned_median.min(), o.ratio_const_over_learned_median.max()))
