from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from env.stochastic_demand import StochasticDemandGenerator, IntersectionTurningConfig


def test_turning_ratio_randomization() -> None:
    generator = StochasticDemandGenerator(seed=42)
    
    base_left = 0.25
    base_straight = 0.50
    base_right = 0.25
    
    left, straight, right = generator.randomize_turning_ratio(
        base_left=base_left,
        base_straight=base_straight,
        base_right=base_right,
        variation_magnitude=0.08,
    )
    
    total = left + straight + right
    assert abs(total - 1.0) < 1e-6, f"Ratios must sum to 1.0, got {total}"
    assert 0.0 <= left <= 1.0, f"Left ratio out of bounds: {left}"
    assert 0.0 <= straight <= 1.0, f"Straight ratio out of bounds: {straight}"
    assert 0.0 <= right <= 1.0, f"Right ratio out of bounds: {right}"
    
    print(f"✓ Turning ratio randomization: ({left:.3f}, {straight:.3f}, {right:.3f})")


def test_arrival_probability_randomization() -> None:
    generator = StochasticDemandGenerator(seed=42)
    
    base_probability = 0.5
    randomized = generator.randomize_arrival_probability(
        base_probability=base_probability,
        variation_range=0.15,
    )
    
    assert 0.0 < randomized < 1.0, f"Randomized probability out of bounds: {randomized}"
    assert 0.35 <= randomized <= 0.65, f"Randomized probability outside expected range: {randomized}"
    
    print(f"✓ Arrival probability randomization: {randomized:.3f}")


def test_flow_conservation() -> None:
    generator = StochasticDemandGenerator(seed=42)
    
    incoming_flow = {
        "N2TL": 100.0,
        "S2TL": 80.0,
        "E2TL": 90.0,
        "W2TL": 110.0,
    }
    
    turning_ratios = {
        "N2TL": (0.25, 0.50, 0.25),
        "S2TL": (0.30, 0.45, 0.25),
        "E2TL": (0.20, 0.55, 0.25),
        "W2TL": (0.25, 0.50, 0.25),
    }
    
    outgoing_flow = generator.ensure_flow_conservation(incoming_flow, turning_ratios)
    
    total_in = sum(incoming_flow.values())
    total_out = sum(outgoing_flow.values())
    
    assert abs(total_in - total_out) < 0.01, f"Flow not conserved: in={total_in}, out={total_out}"
    
    print(f"✓ Flow conservation: input={total_in:.1f}, output={total_out:.1f}")


def test_balanced_turning_ratios() -> None:
    generator = StochasticDemandGenerator(seed=42)
    
    num_directions = 4
    ratios = generator.generate_balanced_turning_ratios(num_directions=num_directions)
    
    assert len(ratios) == num_directions, f"Expected {num_directions} directions, got {len(ratios)}"
    
    for idx, (left, straight, right) in enumerate(ratios):
        total = left + straight + right
        assert abs(total - 1.0) < 1e-6, f"Direction {idx}: ratios sum to {total}, expected 1.0"
    
    print(f"✓ Balanced turning ratios for {num_directions} directions generated")


def main() -> None:
    print("=" * 80)
    print("STOCHASTIC DEMAND GENERATOR TESTS")
    print("=" * 80)
    
    try:
        test_turning_ratio_randomization()
        test_arrival_probability_randomization()
        test_flow_conservation()
        test_balanced_turning_ratios()
        
        print("\n" + "=" * 80)
        print("ALL TESTS PASSED")
        print("=" * 80)
        return
    except AssertionError as exc:
        print(f"\nTEST FAILED: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()