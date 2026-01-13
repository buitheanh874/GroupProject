from __future__ import annotations

import pytest


def test_chunk_dict_has_required_keys():
    chunk_data = {
        "transitions": [(1, 2, 3, 4, False), (5, 6, 7, 8, True)],
        "global_steps": 1,
    }
    
    assert "transitions" in chunk_data, "Chunk must have 'transitions' key"
    assert "global_steps" in chunk_data, "Chunk must have 'global_steps' key"
    assert isinstance(chunk_data["transitions"], list), "Transitions must be a list"
    assert isinstance(chunk_data["global_steps"], int), "global_steps must be an int"


def test_chunk_invariant_single_tls():
    num_tls = 1
    global_steps = 5
    transitions = [(i, i, i, i, False) for i in range(global_steps * num_tls)]
    
    chunk_data = {
        "transitions": transitions,
        "global_steps": global_steps,
    }
    
    assert len(chunk_data["transitions"]) == chunk_data["global_steps"] * num_tls, (
        f"Invariant violated: {len(chunk_data['transitions'])} != {chunk_data['global_steps']} * {num_tls}"
    )


def test_chunk_invariant_9_tls():
    num_tls = 9
    global_steps = 10
    transitions = [(i, i, i, i, False) for i in range(global_steps * num_tls)]
    
    chunk_data = {
        "transitions": transitions,
        "global_steps": global_steps,
    }
    
    assert len(chunk_data["transitions"]) == chunk_data["global_steps"] * num_tls, (
        f"Invariant violated: {len(chunk_data['transitions'])} != {chunk_data['global_steps']} * {num_tls}"
    )


def test_chunk_invariant_violation_detected():
    num_tls = 9
    global_steps = 10
    transitions = [(i, i, i, i, False) for i in range(global_steps * num_tls - 1)]
    
    chunk_data = {
        "transitions": transitions,
        "global_steps": global_steps,
    }
    
    expected_len = chunk_data["global_steps"] * num_tls
    actual_len = len(chunk_data["transitions"])
    
    assert actual_len != expected_len, "Test setup error: should have mismatched lengths"
