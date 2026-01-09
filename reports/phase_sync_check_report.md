# Phase Sync Check Report
## Executive Summary
Phase sync check PASS.
## What was checked
Phase ordering consistency across TLS; NS/EW action semantics via forced greens; controlled links shape.
## Methodology
Collect per-step TLS phase/state via TraCI; build duration-weighted phase signatures; identify main greens; compare ordering across TLS; force NS/EW greens briefly and compare halting vehicles by direction to detect inversions.
## Results
Status: PASS
Reference TLS: J0; main greens: [0, 3]; main order: [3, 0]
## Hard Failures
None
## Soft Failures (Coverage Gate)
None
## Warnings
- ambiguous_tls: ['J0', 'J1', 'J14', 'J17', 'J2', 'J3', 'J4', 'J6', 'J7']
## Semantic Verification Summary
TLS count: 9
Verified count: 0
Verified fraction: 0.00
Ordering mismatches: []
Ordering unknown: []
Inverted TLS: []
Ambiguous TLS: ['J0', 'J1', 'J14', 'J17', 'J2', 'J3', 'J4', 'J6', 'J7']
Skipped semantic: []
require_semantic: False
min_verified_fraction: 0.0
CSV log: reports\phase_sync_log.csv
## Recommendations
No action required.
## How to run
python scripts/check_phase_sync.py --config configs/train_1.yaml --steps 300 --out_dir reports
