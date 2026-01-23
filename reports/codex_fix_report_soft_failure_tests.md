# Code Fix Report — Soft Failure Migration: Test Failures

## Executive Summary
- Fixed unit test regressions tied to invalid mini action spaces (splits=1) and a double time-normalization path that compressed rewards to ~1.0 when transitions were present.

## Root Cause Analysis
- splits=1 failures: Test helpers built SUMOEnv configs with only one split, violating the enforced 15-action design (3 cycles × 5 splits).
- reward=1.0 failure: `_step_legacy` always divided the output of `compute_normalized_reward` by the decision duration, causing a second normalization when transition time was already baked into `t_step`.

## Changes Made
- env/sumo_env.py: Adjusted reward-time normalization to avoid re-dividing transition-inclusive rewards by sim time while preserving legacy normalization when no transitions are configured.
- tests/test_route_pool_selection.py: Test helper now uses the valid 3-cycle/5-split action space.
- tests/test_transition_waiting_flag.py: Updated helper to the 3-cycle/5-split space and fixed interval builder calls to pass the TLS id explicitly.

## Verification
- python -m pytest tests/test_reward_time_1.py -v
- python -m pytest tests/test_route_pool_selection.py -v
- python -m pytest tests/test_transition_waiting_flag.py -v
- python -m pytest tests/ -v --ignore=tests/__pycache__

## Notes / Follow-ups
- Reward normalization now avoids double-scaling when transition phases extend the decision duration; real configs with yellow/all-red timing inherit this corrected behavior.
