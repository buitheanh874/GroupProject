# Semantic Probe (MDP State) Report

## Metadata
- **Timestamp**: 2026-01-10 21:23:45
- **Config**: `configs/train_1.yaml`
- **Route file**: `C:\Users\Dell\GroupProject2\networks\variants\train\bignet_train_seed00042.rou.xml`
- **Git commit**: `b88676d3`
- **Args**: warmup=60, baseline=30, hold=40, min_baseline_queue=5.0

## Summary
- **Status**: PASS
- **TLS count**: 9
- **Repeats**: 1
- **Expected rows**: 9
- **Rows written**: 9
- **Status counts**: {'ambiguous': 5, 'consistent': 4}
- **Vehicle count (min/mean/max)**: 570.87 / 1387.21 / 1734.43

## Results
- **Consistent TLS**: ['J1', 'J3', 'J4', 'J7']
- **Inverted TLS**: []
- **Ambiguous TLS**: ['J0', 'J14', 'J17', 'J2', 'J6']
- **Skipped TLS**: []
- **Error TLS**: []
- **Top reasons**: [('low_queue_signal', 3), ('low_delta', 2)]

## Output Files
- **CSV (timestamped)**: `reports\semantic_probe_state_20260110_212345.csv`
- **CSV (latest)**: `reports\semantic_probe_state_latest.csv`
- **Report (timestamped)**: `reports\semantic_probe_state_20260110_212345.md`
- **Report (latest)**: `reports\semantic_probe_state_latest.md`
