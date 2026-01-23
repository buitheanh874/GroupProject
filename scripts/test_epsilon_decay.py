"""Test epsilon decay implementation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl.parallel_collector_1 import compute_epsilon

# Test multiplier ordering at step 20000
step = 20000
mults = [0.85, 0.95, 1.05, 1.15]
print("Testing multiplier ordering at step 20000:")
eps_vals = []
for m in mults:
    eps = compute_epsilon(step, 0.60, 0.05, 8000, 60000, m)
    eps_vals.append(eps)
    print(f"  mult={m:.2f} -> eps={eps:.4f}")

assert eps_vals == sorted(eps_vals), "Multipliers should give ordered epsilon"
print("[OK] Multiplier ordering verified")

# Test at warmup phase (step 4000)
print("\nTesting at warmup phase (step 4000):")
for m in mults:
    eps = compute_epsilon(4000, 0.60, 0.05, 8000, 60000, m)
    print(f"  mult={m:.2f} -> eps={eps:.4f}")
    # At warmup, eps_base = 0.60, so eps = 0.60 * m
    expected = min(1.0, 0.60 * m)
    assert abs(eps - expected) < 0.001, f"Expected {expected:.4f}, got {eps:.4f}"
print("[OK] Warmup phase verified")

# Full schedule summary
print("\n=== Full Schedule Summary ===")
print(f"eps_start=0.60, eps_end=0.05, warmup=8000, decay=60000")
print(f"Worker multipliers: {mults}")
print("\neps(t) at key points:")
for step in [0, 8000, 38000, 68000, 87000]:
    print(f"\n  global_step={step}:")
    for m in mults:
        eps = compute_epsilon(step, 0.60, 0.05, 8000, 60000, m)
        print(f"    worker mult={m:.2f} -> eps={eps:.4f}")

print("\n[OK] All tests passed!")
