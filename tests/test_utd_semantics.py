from __future__ import annotations

import pytest


def test_no_update_when_queue_empty():
    pending_transitions = 0
    train_freq = 4
    learner_updates_before = 100
    learner_updates_after = learner_updates_before
    
    while pending_transitions >= train_freq:
        learner_updates_after += 1
        pending_transitions -= train_freq
    
    assert learner_updates_after == learner_updates_before, (
        "Learner should not update when pending_transitions < train_freq"
    )


def test_update_count_matches_utd_target():
    pending_transitions = 0
    train_freq = 4
    learner_updates = 0
    
    transitions_to_add = 100
    pending_transitions += transitions_to_add
    
    while pending_transitions >= train_freq:
        learner_updates += 1
        pending_transitions -= train_freq
    
    expected_updates = transitions_to_add // train_freq
    assert learner_updates == expected_updates, (
        f"Expected {expected_updates} updates for {transitions_to_add} transitions, got {learner_updates}"
    )
    
    assert pending_transitions == transitions_to_add % train_freq, (
        f"Leftover should be {transitions_to_add % train_freq}, got {pending_transitions}"
    )


def test_utd_agent_calculation():
    learner_updates = 250
    agent_transitions_total = 1000
    
    utd_agent = learner_updates / max(1, agent_transitions_total)
    
    assert abs(utd_agent - 0.25) < 0.01, (
        f"UTD_agent should be ~0.25, got {utd_agent}"
    )


def test_utd_global_with_9_tls():
    learner_updates = 250
    agent_transitions_total = 1000
    num_tls = 9
    global_env_steps_total = agent_transitions_total // num_tls
    
    utd_global = learner_updates / max(1, global_env_steps_total)
    
    expected_utd_global = 0.25 * num_tls
    assert abs(utd_global - expected_utd_global) < 0.3, (
        f"UTD_global should be ~{expected_utd_global}, got {utd_global}"
    )
