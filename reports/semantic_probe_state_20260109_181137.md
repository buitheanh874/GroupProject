# Semantic Probe (MDP State) Report

## Metadata
- **Timestamp**: 2026-01-09 18:11:37
- **Config**: `configs/train_1.yaml`
- **Route file**: `C:\Users\Dell\GroupProject2\networks\variants\train\bignet_train_seed00042.rou.xml`
- **Git commit**: `06bbc3dd`
- **Args**: warmup=60, baseline=30, hold=40, min_baseline_queue=5.0

## Summary
- **Status**: FAIL
- **TLS count**: 9
- **Repeats**: 3
- **Expected rows**: 27
- **Rows written**: 27
- **Status counts**: {'ambiguous': 9, 'consistent': 15, 'inverted': 3}
- **Vehicle count (min/mean/max)**: 638.73 / 2090.40 / 2873.47

## Results
- **Consistent TLS**: ['J1', 'J14', 'J17', 'J3', 'J6']
- **Inverted TLS**: ['J7']
- **Ambiguous TLS**: ['J0', 'J2', 'J4']
- **Skipped TLS**: []
- **Error TLS**: []
- **Top reasons**: [('low_delta', 9), ('inverted_response', 3)]

## Output Files
- **CSV (timestamped)**: `reports\semantic_probe_state_20260109_181137.csv`
- **CSV (latest)**: `reports\semantic_probe_state_latest.csv`
- **Report (timestamped)**: `reports\semantic_probe_state_20260109_181137.md`
- **Report (latest)**: `reports\semantic_probe_state_latest.md`
