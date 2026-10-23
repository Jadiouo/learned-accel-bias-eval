# Coverage (FULL)

results_dir: `<REPO>/euroc_full/results_vicon`

runs_index rows: 684; ok: 684; failed: 0; expected total: 684

| method | expected | ok | failed | not run |
|---|---|---|---|---|
| constant_perseq | 6 | 6 | 0 | 0 |
| linear_stats | 6 | 6 | 0 | 0 |
| kf | 6 | 6 | 0 | 0 |
| pf | 6 | 6 | 0 | 0 |
| tcn | 60 | 60 | 0 | 0 |
| transformer_s | 60 | 60 | 0 | 0 |
| transformer_m | 60 | 60 | 0 | 0 |
| transformer_l | 60 | 60 | 0 | 0 |
| mamba | 60 | 60 | 0 | 0 |
| m1_cnn | 60 | 60 | 0 | 0 |
| m2 | 60 | 60 | 0 | 0 |
| m3_lstm | 60 | 60 | 0 | 0 |
| m4 | 60 | 60 | 0 | 0 |
| m5 | 60 | 60 | 0 | 0 |
| m6 | 60 | 60 | 0 | 0 |

Complete-case deep-method set (>= 10 ok seeds on every sequence): ['tcn', 'transformer_s', 'transformer_m', 'transformer_l', 'mamba', 'm1_cnn', 'm2', 'm3_lstm', 'm4', 'm5', 'm6']
Excluded from cross-method statistics: []

ok seeds per deep method x sequence:

```
eval_sequence  V1_01  V1_02  V1_03  V2_01  V2_02  V2_03
method                                                 
tcn               10     10     10     10     10     10
transformer_s     10     10     10     10     10     10
transformer_m     10     10     10     10     10     10
transformer_l     10     10     10     10     10     10
mamba             10     10     10     10     10     10
m1_cnn            10     10     10     10     10     10
m2                10     10     10     10     10     10
m3_lstm           10     10     10     10     10     10
m4                10     10     10     10     10     10
m5                10     10     10     10     10     10
m6                10     10     10     10     10     10
```
