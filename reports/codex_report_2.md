Summary
- Locked the SUMO action space to the canonical 5-5-5 order and tightened validation around cycles, splits, and action tables.
- Synced docs, helpers, and templates to the 60/90/120 defaults and protected manifests against bad inputs.

Changes by file
- env/sumo_env.py: canonicalized multi-TLS action_table builds to cycle-then-split order with duplicate/missing checks.
- scripts/validation.py: enforced non-empty splits, 3 cycles/5 splits for 12D, and full cycle/split coverage that must match configured splits.
- scripts/common.py: unified cycle_options as the single source for allowed cycles and added a min_green_sec alias.
- scripts/routes_1.py: aborts if train/eval route selections overlap while keeping non-empty route validation.
- tests/test_action_map_1.py: added coverage that shuffled action_table entries still map to the fixed 0-14 ordering.
- tests/test_templates_validate.py: template checks now use 60/90/120 cycles with the five default splits.
- scripts/cycle_distribution_logger.py: default cycles moved to 60/90/120.
- docs/README.md, README.md, docs/integration_guide.md, docs/audit.md: documented the 5-5-5 mapping and refreshed defaults.
- reports/antigravity_run_report.md removed as outdated.

Validation added
- For state_dim=12, configs must declare exactly three cycle options and five splits; action tables must cover every cycle/split pair and match the split list.
- Manifest generation now fails fast when train and eval route lists overlap; route files remain guarded by non-empty checks; min_green_sec is accepted as an alias.

Tests added/updated
- tests/test_action_map_1.py: ensures canonical action ids even when action_table order is shuffled.
- tests/test_templates_validate.py: validates templates against 60/90/120 cycles and five fixed splits.

Files removed
- reports/antigravity_run_report.md (obsolete 30/60/90 report; `rg antigravity_run_report` shows no remaining references).

Manual commands for user
- a) `python -m scripts.collect_norm_stats --config configs/train_1.yaml --episodes 50 --out configs/norm_1.json`
- b) `python -m pytest -q`
- c) `python scripts/train.py --config configs/train_1.yaml --episodes 10`
- d) `python scripts/eval.py --config configs/eval_1.yaml --controller fixed --runs 1` (repeat with `--controller max_pressure` if needed)
- e) `python scripts/train.py --config configs/train_1.yaml --episodes 500`

Expected outcomes
- Action ids stay fixed: 0-4 = cycle 60 (splits 0.30/0.70 -> 0.70/0.30), 5-9 = cycle 90, 10-14 = cycle 120; fixed-time or max-pressure selection should align with this space.
- Reward timing uses TraCI `decision_duration_sec` per decision; when `reward_time_normalize` is true, rewards divide by that duration (see info logs).
- `scripts/routes_1.py` should emit `networks/variants/train/manifest_1.txt` and `networks/variants/eval/manifest_1.txt` with no overlap; route loader will fail fast on empty or missing route files.
- Norm stats command should write `configs/norm_1.json`; pytest should pass; train/eval runs should draw routes from manifest_1.txt and log `cycle_sec`/`decision_duration_sec` consistent with 60/90/120 ordering.
