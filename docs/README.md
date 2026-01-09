# GroupProject

Multi-TLS hub-and-spoke setup with shared Dueling DQN. See `docs/README.md` for config keys and `configs/train_hub_spoke_demo.yaml` for the 12D state/action-table sample. Multi-node assets belong under `networks/hub_spoke/`.

- `scenario_calibration` (YAML) auto-injects `env.sumo.vehicle_weights` for PCU-weighted rewards; set `force_calibration_overrides: true` to override user weights, or keep user weights by default.
- Action table schema prefers `rho_ns`; legacy `ns_ratio` is still accepted and normalized to `rho_ns` automatically.
- Route pools support manifests/relative paths; use manifests under `train/` or `eval/` when wiring generated variants.
- Generated Hanoi variants emit SUMO-compatible `flows_*.xml` (vTypes + leaf `<flow>` nodes) and `turns_*.xml` (edgeRelation-only, normalized to 6 decimals).
- See `docs/HANOI_SETUP.md` for calibration/variant wiring and `docs/MDP_COMPLIANCE.md` for MDP compliance notes.

## Action Space (5-5-5)
- Cycles fixed to [60, 90, 120] with five splits [(0.30,0.70), (0.40,0.60), (0.50,0.50), (0.60,0.40), (0.70,0.30)].
- Action ids: 0-4 map to cycle 60 with the split order above, 5-9 to cycle 90, 10-14 to cycle 120.
- `reward_time_normalize` uses TraCI simulation time per decision when enabled.

## Multi-TLS Coverage
- Supports arbitrary `tls_ids` (N TLS). Verified with hub-spoke (5 TLS) and BIGNET template (9 TLS).
- New templates: `configs/train_bignet_9tls.yaml`, `configs/eval_bignet_9tls.yaml` (use `auto_tls_ids: true` to discover IDs from `networks/BIGNET.net.xml`).
- Quick smoke commands: `python scripts/train.py --config configs/train_bignet_9tls.yaml --episodes 5` and `python scripts/eval.py --config configs/eval_bignet_9tls.yaml --controller fixed --runs 1` (fill route/lane mappings first).

## Deadlock/Gridlock Shaping (New Direction)

Long-horizon training with deadlock-based reward shaping for realistic traffic control evaluation:

- **Simulation Settings**: `--time-to-teleport=300`, `max_sim_seconds=1800` (30-minute episodes)
- **Training**: Deadlock shaping enabled with early-warning penalty (no_arrival_sec >= 30s) and hard deadlock trigger (no_arrival_sec >= 150s). Episode terminates on deadlock trigger. Teleport penalty disabled (`teleport_penalty_lambda=0.0`).
- **Teleport-under-congestion**: Treated as failure event if teleport occurs with congestion evidence (queue/occupancy thresholds).
- **Evaluation**: Deadlock shaping disabled to allow full episode completion. Results based on corrected KPIs (teleport_rate, avg_wait_time_corr, throughput_corr).
- **Config files**: `configs/train_bignet_9tls_long_tele300.yaml`, `configs/eval_bignet_9tls_long_tele300.yaml`
- **CSV columns**: New additive columns `deadlock_triggered`, `deadlock_reason`, `deadlock_no_arrival_sec` for analysis.
