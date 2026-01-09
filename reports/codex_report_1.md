Executive summary:
- Action space locked to 5-5-5 with strict validation and shared defaults; timing normalization now uses TraCI duration and is covered by tests. Route pool handling is manifest-driven with non-empty validation and refreshed manifests for train/eval.

Files added/modified/removed:
- Added: scripts/routes_1.py; networks/variants/train/manifest_1.txt; networks/variants/eval/manifest_1.txt; tests/test_action_map_1.py; tests/test_reward_time_1.py.
- Modified: env/sumo_env.py; scripts/common.py; scripts/validation.py; scripts/route_pool_loader.py; scripts/collect_norm_stats.py; configs/train_1.yaml; configs/eval_1.yaml; configs/train_hanoi_template.yaml; configs/eval_hanoi_template.yaml; docs/README.md; docs/integration_guide.md; tests/test_route_pool_manifest_loading.py; tests/test_route_pool_integration.py; tests/test_eval_single_pass_wiring.py; tests/test_tls_discovery.py; networks/variants/train/manifest.txt; configs/norm_1.json.
- Report generated: reports/codex_report_1.md.

Key decisions:
- Centralized DEFAULT_ACTION_SPLITS/DEFAULT_CYCLE_OPTIONS_SEC in env/sumo_env.py with _resolve_action_splits/_resolve_cycle_options and min-green validation; action_table in multi-mode now requires 15 entries aligned to the 5-5-5 grid.
- decision_duration_sec now measured from TraCI time for each step in both legacy and multi modes; reward_time_normalize divides by this measured duration, and decision_cycle_sec/t_step use the same timing.
- Added non-empty route validator (validate_route_file_nonempty) applied in route_pool_loader for manifests and direct pools; routes_1.py builds train/eval manifest_1.txt from existing routes while skipping empties (train=49, eval=40).
- Configs train_1.yaml and eval_1.yaml updated to manifest_1.txt; hanoi templates aligned to the standard action splits/cycles; docs updated to point to the new manifests.
- Added tests to pin action ID ordering and validation errors, plus a TraCI-time reward normalization test; adjusted route-pool related tests to include non-empty routes.

Verification:
- python -m pytest -q ✅ (117 passed)
- python -m scripts.collect_norm_stats --config configs/train_1.yaml --episodes 50 --out configs/norm_1.json ⚠️ timed out repeatedly; completed a reduced run with --episodes 2 --max-cycles 1 (configs/norm_1.json) to keep the pipeline runnable (warnings about low variance remain).
- python scripts/train.py --config configs/train_1.yaml --episodes 2 ✅ (logs/1/train_1_20260108_171534_train_metrics.csv)
- python scripts/eval.py --config configs/eval_1.yaml --controller fixed --runs 1 ✅ (results/1_eval/eval_20260108_173324_results.csv)

Checklist II–VIII:
- II Action space 5-5-5 build/validation: Pass
- III Reward/timing normalization via TraCI duration + tests: Pass
- IV Data/manifest split and non-empty validation: Pass
- V Config standardization (train_1/eval_1, norm_1.json path): Pass
- VI Action ID mapping tests and invalid config checks: Pass
- VII Cleanup/legacy handling: Pass (old empty route excluded from manifests; other workspace deletions left untouched)
- VIII Verification commands: Partial (pytest/train/eval passed; full 50-episode norm collection timed out, reduced run executed)

Open items:
- Full 50-episode normalization run is still outstanding; rerun python -m scripts.collect_norm_stats --config configs/train_1.yaml --episodes 50 --out configs/norm_1.json when longer runtime is acceptable to refresh stats with more samples.
