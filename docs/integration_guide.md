# Integration Guide - SUMO Train/Eval (Hub-Spoke 5 TLS, BIGNET 9 TLS)

This guide is synced to the current codebase and keeps the original Claude audit summary for context (see Appendix). It covers single TLS (`env.sumo.tls_id`), multi TLS (`env.sumo.tls_ids`), auto discovery (`tls_ids: "auto"` or `auto_tls_ids: true`), the 5-TLS hub-spoke template, and the 9-TLS BIGNET template.

## Current Status / Implementation Notes
- In repo now: `scripts/sumo_network_tools.py` (TLS discovery via `extract_tls_ids`), `tests/test_tls_discovery.py`, multi-TLS configs (`configs/train_bignet_9tls.yaml`, `configs/eval_bignet_9tls.yaml`, hub-spoke templates), multi-TLS support in `scripts/train.py`/`scripts/eval.py` and `env/sumo_env.py` (lane_groups_by_tls validation, state_dim checks, cycle masking helpers).
- Auto TLS: set `env.sumo.tls_ids: "auto"` or `env.sumo.auto_tls_ids: true` to pull IDs (order-preserving) from `net_file` via `extract_tls_ids`. `center_tls_id` must be inside `tls_ids` (tests enforce).
- Placeholders/gaps: BIGNET `lane_groups_by_tls` and `downstream_links` use `REPLACE_*`; `networks/BIGNET.rou.xml` is not provided; hub-spoke SUMO assets (`networks/hub_spoke/hub_spoke.*`) and route pool manifests under `networks/variants/...` are absent; normalization stats files need to be generated.
- Acceptance criteria: configs load through `scripts.common.build_env`; TLS discovery returns 9 IDs for BIGNET; lane_groups_by_tls populated per TLS; downstream_links set or occupancy disabled; `pytest tests/test_tls_discovery.py` passes; train/eval commands run with real SUMO assets.
- Known TODOs: provide real routes for both scenarios, fill lane mappings and downstream links before enabling occupancy, set `eval.model_path` for RL runs, collect normalization stats.

## Prerequisites
- Python 3.9+ with PyTorch, NumPy, and PyYAML available (use the existing `.venv` or install these packages).
- SUMO and TraCI installed; `sumo`/`sumo-gui` (and `netedit` if desired) on PATH. `scripts/generate_jtr_data.py` calls `python -m sumolib.tools.generateRoutes`.
- Run commands from repo root (`C:/Users/Dell/GroupProject`); paths below are relative.

## Quick Start 1 - Hub-spoke (5 TLS)
- Prepare assets: place `networks/hub_spoke/hub_spoke.net.xml` and `networks/hub_spoke/hub_spoke.rou.xml`; create the route pool manifests referenced in configs (`train.route_pool_manifest: networks/variants/train/manifest_1.txt`, `eval.route_pool_manifest: networks/variants/eval/manifest_1.txt`) or set those fields to null if using a single route file.
- Train smoke (uses `configs/train_hub_spoke_demo.yaml`, logs to `logs/hub_spoke_demo`, models to `models/hub_spoke_demo`):
```bash
python scripts/train.py --config configs/train_hub_spoke_demo.yaml --episodes 5
```
- Eval smoke (fixed controller to avoid model_path requirement; writes to `results/hub_spoke_eval`):
```bash
python scripts/eval.py --config configs/eval_hub_spoke.yaml --controller fixed --runs 1
```
- For RL eval, set `eval.model_path` in the config or pass `--model_path models/hub_spoke_demo.pt` and use `--controller rl` or `--controller all`.

## Quick Start 2 - BIGNET (9 TLS)
- Files in repo: `networks/BIGNET.net.xml`, `configs/train_bignet_9tls.yaml`, `configs/eval_bignet_9tls.yaml`. Provide `networks/BIGNET.rou.xml`, fill all `REPLACE_*` lane IDs in `lane_groups_by_tls`, and add `downstream_links` before enabling occupancy.
- Auto TLS options: set `env.sumo.tls_ids: "auto"` or `env.sumo.auto_tls_ids: true` to read IDs from the net file. Manual list defaults to `["J0", "J1", "J14", "J17", "J2", "J3", "J4", "J6", "J7"]`.
- Train smoke (logs/models/results under `logs/models/results/bignet_9tls`):
```bash
python scripts/train.py --config configs/train_bignet_9tls.yaml --episodes 5
```
- Eval smoke (fixed controller; set `eval.model_path` plus `--controller rl` when ready):
```bash
python scripts/eval.py --config configs/eval_bignet_9tls.yaml --controller fixed --runs 1
```
- TLS discovery from the net file:
```bash
python - <<'PY'
from scripts.sumo_network_tools import extract_tls_ids
print(extract_tls_ids("networks/BIGNET.net.xml"))
PY
```

## Configuration Reference
- `env.sumo.net_file`, `env.sumo.route_file`, `env.sumo.additional_files`: must exist; `build_env` raises `FileNotFoundError` if missing.
- `env.sumo.tls_id` (single), `env.sumo.tls_ids` (list or `"auto"`), `env.sumo.auto_tls_ids: true`: tls_ids must be non-empty and unique; auto modes load IDs from the network (order preserved). `env.sumo.center_tls_id` defaults to the first tls_id and must be in tls_ids.
- `state_dim`: defaults to 12 when multiple tls_ids or when `action_table` is provided; must be 12 for multi-TLS; allowed values are 4 or 12.
- `lane_groups_by_tls` (dict) required when tls_ids has multiple entries; each TLS must define non-empty `lanes_ns_ctrl` and `lanes_ew_ctrl`; optional `lanes_right_turn_slip_*` and `approach_lanes` must not overlap controlled lanes. Single-TLS fallback uses `lane_groups`.
- `phase_program`: `ns_green` and `ew_green` required; `ns_yellow`/`ew_yellow` needed if `yellow_sec > 0`; `all_red` needed if `all_red_sec > 0`.
- `action_splits` (list of `[rho_ns, rho_ew]`) vs `action_table` (items with `cycle_sec`, `rho_ns`/`ns_ratio`, optional `rho_ew`); `allowed_cycles_sec` defaults to `[60, 90, 120]` and cannot be empty; `g_min_sec` default 5 enforces minimum green. When `state_dim = 12` and `action_table` is empty, actions auto-generate from `action_splits` across allowed cycles.
- `route_pool_manifest` / `route_pool` under `train` or `eval`: manifest takes priority; paths are resolved relative to the manifest and project root, globs supported when using `route_pool`. Resolved routes are injected into `env.sumo.route_pool`.
- `normalization`: supply `mean`/`std` vectors of length `state_dim` or set `file` to a JSON containing `mean`/`std`. Required when `normalize_state: true`; disabled (`mean=[0...], std=[1...]`) when `normalize_state: false`.
- `downstream_links` plus `enable_downstream_occupancy`: occupancy is only used for `center_tls_id`; when enabled you must provide N/E/S/W links or the config fails fast (ValueError). Set `enable_downstream_occupancy: false` until links are confirmed.
- `baseline.fixed_action_id`: fixed action used by baseline controllers and normalization collection.

## Recommended Workflow (Train/Eval)
- Validate configs before running:
```bash
pytest tests/test_tls_discovery.py -q
python - <<'PY'
from scripts.common import resolve_tls_ids_from_sumo_cfg
import yaml, pathlib
cfg = yaml.safe_load(open("configs/train_bignet_9tls.yaml"))
tls, center = resolve_tls_ids_from_sumo_cfg(cfg["env"]["sumo"], pathlib.Path(cfg["env"]["sumo"]["net_file"]))
print("tls_ids", tls, "center", center)
PY
```
- Hub-spoke smoke: after assets and route manifests exist, run `python scripts/train.py --config configs/train_hub_spoke_demo.yaml --episodes 5`, then `python scripts/eval.py --config configs/eval_hub_spoke.yaml --controller fixed --runs 1`.
- BIGNET workflow:
  1) Replace `REPLACE_*` lane IDs and downstream links; add `networks/BIGNET.rou.xml`.
  2) Collect stats (50+ samples recommended):
```bash
python scripts/collect_norm_stats.py --config configs/train_bignet_9tls.yaml --episodes 10 --out configs/norm_stats_bignet_9tls.json
```
  3) Train full run:
```bash
python scripts/train.py --config configs/train_bignet_9tls.yaml --episodes 300
```
  4) Eval:
```bash
python scripts/eval.py --config configs/eval_bignet_9tls.yaml --controller all --model_path models/bignet_9tls.pt --runs 3
```
- Route generation option for BIGNET (uses SUMO tools):
```bash
python scripts/generate_jtr_data.py --net-file networks/BIGNET.net.xml --output-route networks/BIGNET.rou.xml --seed 42 --volume-scale 1.0
```

## Troubleshooting
- `tls_ids must be non-empty` / duplicates / center TLS error: ensure `env.sumo.tls_ids` is a unique list or set `auto_tls_ids: true`; set `center_tls_id` to one of them.
- `When specifying multiple tls_ids, state_dim must be 12`: keep `state_dim: 12` for multi-TLS configs.
- `lane_groups_by_tls missing definitions` or `lanes_*_ctrl must not be empty`: define non-empty lanes for every TLS; remove overlaps with slip lanes.
- SUMO file errors (`Network file not found`, `Route file not found`): keep paths relative to repo root and create missing assets (`networks/hub_spoke/hub_spoke.*`, `networks/BIGNET.rou.xml`).
- Simulation ends immediately or no vehicles: route file is empty or `terminate_on_empty: true`; provide real routes or disable terminate_on_empty while debugging.
- Eval RL failure: `model_path is required for RL evaluation` appears when using `--controller rl/all` without a checkpoint.
- Downstream occupancy validation: provide N/E/S/W edges in `downstream_links` for `center_tls_id` or set `enable_downstream_occupancy: false`; missing links now raise ValueError (no zero-fill fallback).

## Appendix: Claude Audit Notes (kept for traceability)
- Artifacts noted as ready: `scripts/sumo_network_tools.py` (TLS discovery helper), `tests/test_tls_discovery.py`, `configs/train_bignet_9tls.yaml`, `configs/eval_bignet_9tls.yaml`, placeholder `networks/BIGNET.rou.xml`; earlier draft suggested extra helpers in `scripts/common.py` (current code already handles auto TLS via `resolve_tls_ids_from_sumo_cfg` plus `extract_tls_ids`).
- Historical integration reminders from Claude: copy the above files, add typing imports if new helpers are introduced, and rerun `pytest tests/test_tls_discovery.py`.
- Post-integration tasks: replace all `REPLACE_*` lane IDs and downstream links per TLS; fill BIGNET routes; collect normalization stats via `scripts/collect_norm_stats.py`; generate routes with `scripts/generate_jtr_data.py` or manual edits.
- Validation checklist (from Claude): configs load; TLS discovery returns 9 IDs; lane groups defined for all TLS; downstream links mapped for center; normalization stats collected; smoke train/eval succeed.
- Common issues noted: missing module imports when not running from project root; lane_groups placeholders triggering validation; SUMO failing to open missing network files.
- Original examples (adapted): auto TLS via `env.sumo.auto_tls_ids: true`; validation helper using `extract_tls_ids` to confirm 9 TLS; smoke commands mirror the Quick Start sections above.
- Key concepts preserved: multi-agent parameter sharing across TLS, 12D state `[queues, waiting, occupancy]` (occupancy only for center, zero elsewhere), action space built from cycle/split combinations.
- Summary of needs: provide real BIGNET lanes/routes, run tests, and perform smoke train/eval once assets are in place.
