Goal: Add "Parallel Collector + Single Learner" for DQN + SUMO (single-thread CPU) to speed up experience collection with multiple processes while learner updates a single model.

Principles:
- Feature only enabled when config.parallel.enabled = true. Default false keeps current training unchanged.
- No migration to other frameworks (no RLlib/SB3/Tianshou). No major architecture refactor.
- Prefer new script scripts/train_parallel.py instead of modifying scripts/train.py.
- SUMO port must be unique per worker_id: port = base_port + worker_id.
- Avoid multiprocessing errors: no Queue.empty/qsize for flow control; use get(timeout)/get_nowait and handle Empty/Full.
- Reduce IPC overhead: collector sends transitions in chunks (e.g. 256 steps) instead of per step.
- Minimal weight sync: learner broadcasts weights periodically for collectors to use new policy.
- CPU safe defaults: each process limits threads (OMP/MKL/torch) = 1 to avoid oversubscription.

New config (additive):
parallel:
  enabled: false
  num_actors: 4
  base_port: 8813
  base_seed: 42
  chunk_size: 256
  queue_max_chunks: 200
  sync_every_updates: 100
  epsilon_base: 0.2
  epsilon_worker_delta: 0.02

Completion criteria:
- scripts/train_parallel.py runs with --dry-run without SUMO, no process spawn.
- Can start collectors and learner (full SUMO not required in CI), code path correct with shutdown handling.
- pytest -q PASS.
- Report reports/parallel_collector_1.md has prepended entry.
