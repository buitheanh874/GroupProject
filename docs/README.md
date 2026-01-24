# GroupProject - Traffic Signal Control RL

Multi-TLS (9 TLS) BIGNET network with shared Dueling DQN. Config-driven 5-5-5 action space.

## Project Structure

-   `configs/` – YAML configs for training and evaluation
-   `env/` – SUMO environment wrapper (`sumo_env.py`)
-   `rl/` – RL agent (Dueling DQN)
-   `scripts/` – Training, evaluation, and utility scripts
-   `networks/` – `BIGNET.net.xml` and route variants

## Action Space (5-5-5)

15 discrete actions = 3 cycles × 5 splits.

| Cycles (sec) | Split Ratios (NS/EW)                                                |
| ------------ | ------------------------------------------------------------------- |
| 60, 90, 120  | (0.30/0.70), (0.40/0.60), (0.50/0.50), (0.60/0.40), (0.70/0.30) |

-   Action IDs 0-4 → cycle 60; 5-9 → cycle 90; 10-14 → cycle 120.
-   `reward_time_normalize: true` divides reward by decision duration.

## TLS Configuration

-   `tls_ids`: List of controlled intersections (e.g., `["J0", "J1", ..., "J17"]`).
-   `lane_groups_by_tls`: Per-TLS `lanes_ns_ctrl`, `lanes_ew_ctrl`, `approach_lanes`.
-   `tls_phase_overrides`: Corrects inverted NS/EW semantics for specific TLS (e.g., J2, J14).

## Normalization

Collect raw state statistics before training:

```bash
python scripts/collect_norm_stats.py --config configs/train_bignet_short.yaml --episodes 10 --out configs/norm_stats.json
```

Config points to the generated file:

```yaml
normalization:
  file: configs/norm_stats.json
```

## Curriculum Learning

Training can use curriculum phases with increasing demand:

```yaml
curriculum:
  enabled: true
  phases:
    - name: "phase1_warmup"
      episodes: 30
      demand_scale: 0.50
      route_pool_manifest: networks/variants/train/manifest_scale50.txt
    # ... more phases
```

Each phase uses a different route manifest.

## Route Management

-   Route files: `.rou.xml` format (SUMO-compatible).
-   Manifests: `.txt` files listing one route file per line.
-   `route_pool_loader.py` resolves manifest to a pool of `.rou.xml` files.

## Quick Start

**Training:**

```bash
python scripts/train.py --config configs/train_bignet_short.yaml
```

**Evaluation:**

```bash
python scripts/eval.py --config configs/eval_bignet_9tls.yaml --controller fixed --runs 5
```

## Deadlock/Gridlock Shaping (Optional)

Experimental:

-   `deadlock_no_arrival_sec`, `deadlock_penalty`, `terminate_on_deadlock`.
-   Teleport penalty: `teleport_penalty_lambda`.

## Utility Scripts

-   `scripts/doctor.py` – Environment health check.
-   `scripts/check_phase_sync.py` – Verify phase semantics.
-   `scripts/semantic_probe_state.py` – State vector diagnostics.

## See Also

-   [MDP_COMPLIANCE.md](MDP_COMPLIANCE.md) – MDP notes.
