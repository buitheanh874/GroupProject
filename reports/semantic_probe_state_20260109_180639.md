# Semantic Probe (MDP State) Report

## Metadata
- **Timestamp**: 2026-01-09 18:06:39
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
- **Status counts**: {'ambiguous': 9, 'consistent': 9, 'inverted': 9}
- **Vehicle count (min/mean/max)**: 638.73 / 2053.32 / 2746.37

## Results
- **Consistent TLS**: ['J1', 'J17', 'J6']
- **Inverted TLS**: ['J14', 'J2', 'J3']
- **Ambiguous TLS**: ['J0', 'J4', 'J7']
- **Skipped TLS**: []
- **Error TLS**: []
- **Top reasons**: [('low_delta', 9), ('inverted_response', 9)]

## Output Files
- **CSV (timestamped)**: `reports\semantic_probe_state_20260109_180639.csv`
- **CSV (latest)**: `reports\semantic_probe_state_latest.csv`
- **Report (timestamped)**: `reports\semantic_probe_state_20260109_180639.md`
- **Report (latest)**: `reports\semantic_probe_state_latest.md`
