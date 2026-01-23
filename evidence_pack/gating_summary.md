# Gating Summary (t=1000, horizon=2000)

| demand | controller | runs | comp_mean | tele_mean | no_arrival_max | slope_last300s | rec |
|---|---|---|---|---|---|---|---|
| 500 | fixed | 2 | 1.0000 | 0.0000 | 13.5 | 0.0000 | TRAIN_SAFE |
| 500 | max_pressure | 2 | 1.0000 | 0.0000 | 13.5 | 0.0000 | TRAIN_SAFE |
| 750 | fixed | 2 | 0.9759 | 0.0001 | 13.0 | -46.3421 | TRAIN_SAFE |
| 750 | max_pressure | 2 | 0.9759 | 0.0001 | 13.0 | -46.3421 | TRAIN_SAFE |
| 1000 | fixed | 2 | 0.9146 | 0.0000 | 5.5 | -29.7632 | EVAL_ONLY |
| 1000 | max_pressure | 2 | 0.9157 | 0.0001 | 5.5 | -30.2895 | EVAL_ONLY |