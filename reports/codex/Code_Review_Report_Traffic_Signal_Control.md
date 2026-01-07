# Code Review Report — Traffic Signal Control

## 2026-01-06 22:52
### Executive Summary
- Fixed CRITICAL-2, CRITICAL-3, CRITICAL-4, CRITICAL-5 and HIGH-1, HIGH-2, HIGH-5; marked CRITICAL-1, HIGH-3, HIGH-4 as false positives.
- Hardened SUMO start, route pool determinism, gamma validation, normalization efficiency, and eval action validation; centralized repo root resolution.

### Detailed Changes
- CRITICAL-1 Memory leak in training loop — Status: False Positive  
  - Files: rl/replay_buffer.py, scripts/train.py  
  - Reasoning: ReplayBuffer uses fixed-capacity preallocated arrays; training metrics flushed per episode via csv_file.flush(); no unbounded lists retained.
  - Tests/Commands: python -m compileall ., pytest -q.
- CRITICAL-2 Route pool selection reseeds shared RNG — Status: Fixed  
  - Files: env/sumo_env.py, tests/test_route_pool_selection.py  
  - Change: Use per-call random.Random(seed) for deterministic selection without mutating env RNG; added tests for determinism and RNG independence.  
  - Tests/Commands: python -m compileall ., pytest -q.
- CRITICAL-3 TraCI connection failure handling — Status: Fixed  
  - Files: env/sumo_env.py  
  - Change: Added retry loop with exponential backoff and free port selection; closes partial connections on failure.  
  - Tests/Commands: python -m compileall ., pytest -q.
- CRITICAL-4 Dangerous float comparison — Status: Fixed  
  - Files: env/stochastic_demand.py, scripts/validation.py  
  - Change: Introduced named tolerance and math.isclose for ratio validation.  
  - Tests/Commands: python -m compileall ., pytest -q.
- CRITICAL-5 Time-aware gamma computation — Status: Fixed  
  - Files: rl/agent.py, tests/test_agent_gamma.py  
  - Change: Validate positive t_ref/t_step, compute gamma via gamma_base ** (t_step / t_ref), added monotonicity and guard tests.  
  - Tests/Commands: python -m compileall ., pytest -q.
- HIGH-1 State normalization efficiency — Status: Fixed  
  - Files: env/normalization.py, tests/test_normalization_efficiency.py  
  - Change: In-place normalization and clipping on copied array to reduce allocations; verified dtype/shape preservation.  
  - Tests/Commands: python -m compileall ., pytest -q.
- HIGH-2 Missing action validation in eval — Status: Fixed  
  - Files: scripts/eval.py, tests/test_eval_wiring_smoke.py  
  - Change: Validate fixed_action_id bounds against resolved action space; keep single model load path.  
  - Tests/Commands: python -m compileall ., pytest -q.
- HIGH-3 waiting_total exponent risk — Status: False Positive  
  - Files: env/mdp_metrics.py  
  - Reasoning: waiting_total clamps exponent to >=1 and not used to alter default reward without explicit enable.
  - Tests/Commands: python -m compileall ., pytest -q.
- HIGH-4 Metrics file robustness — Status: False Positive  
  - Files: scripts/train.py  
  - Reasoning: Metrics written via csv.DictWriter with flush after each episode; no in-memory accumulation beyond per-episode losses.
  - Tests/Commands: python -m compileall ., pytest -q.
- HIGH-5 Hardcoded repo root path — Status: Fixed  
  - Files: scripts/repo_root.py, multiple scripts/*  
  - Change: Added repo root resolver using .git/pyproject detection; replaced Path(__file__).parents[1] across evaluation and utility scripts.  
  - Tests/Commands: python -m compileall ., pytest -q.

### Commands Run
- python -m compileall .
- pytest -q

### Risk & Follow-ups
- SUMO start retry/backoff not integration-tested with a live SUMO server.
- Route pool tests are unit-level; real route files still expected to exist at runtime.
- Run report aligned to current HEAD 038e5e5755fc88addc96f6236fda3c844fe10722.
