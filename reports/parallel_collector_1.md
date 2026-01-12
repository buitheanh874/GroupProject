Run: 2026-01-12 18:31 local
Goal: parallel collector + single learner for DQN + SUMO
Discovery:
- files_read: rl/utils.py, rl/agent.py, rl/replay_buffer.py, env/sumo_env.py, scripts/common.py, configs/train_1.yaml
- key_findings: load_yaml_config in rl/utils.py L33, build_env/build_agent in scripts/common.py, DQNAgent.online_net is network, ReplayBuffer.push(state,action,reward,next_state,done,gamma), sample(batch_size,device), worker_id/base_port already in SumoEnvConfig
Changes:
- added: specs/parallel_collector_1.md, configs/train_parallel_smoke_1.yaml, rl/parallel_collector_1.py, scripts/train_parallel.py, tests/test_train_parallel_dry_run_1.py
- modified: none (env/sumo_env.py already had worker_id/base_port from prior session)
Commands:
- pytest tests/test_train_parallel_dry_run_1.py -q: PASS (2 passed)
- dry_run: PASS (python scripts/train_parallel.py --config configs/train_parallel_smoke_1.yaml --dry-run)
Notes:
- Port uniqueness: port = base_port + worker_id, already implemented in SumoEnvConfig._get_free_port
- Queue handling: no empty/qsize, uses get(timeout)/get_nowait with Empty/Full handling
- Weight sync: learner broadcasts every sync_every_updates, weight_queue maxsize=1 with drain-and-load pattern
- Collector uses same action_id for all TLS to satisfy cycle_sec matching requirement in multi-agent mode
- Thread limits: OMP/MKL/torch set to 1 in collector process
