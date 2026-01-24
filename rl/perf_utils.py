"""
Performance optimization utilities for RL training pipeline.

This module contains:
- PerformanceConfig: Feature flags for toggling optimizations
- TransitionCounters: Transition-level accounting (produced vs consumed)
- TimingBreakdown: Timing instrumentation for bottleneck analysis
- IntervalLogger: Log at intervals instead of per-step
- GoldenTraceRecorder: Record decision-step data for invariance testing

All optimizations are SAFE (do not change MDP/algorithm semantics).
"""
from __future__ import annotations

import time
import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from multiprocessing import Value
import numpy as np


@dataclass
class PerformanceConfig:
    """Feature flags for performance optimizations."""
    enable_all_optimizations: bool = False
    
    # Individual toggles (only apply if enable_all is True)
    disable_sumo_logs: bool = True          # Opt 1: SUMO output suppression
    use_traci_subscriptions: bool = False   # Opt 2: TraCI batch subscriptions (off by default)
    subscribe_scalar_only: bool = True      # Opt 2b: Scalar vars only (safer)
    subscribe_with_id_list: bool = False    # Opt 2c: ID list (only if MDP needs it)
    use_packed_transitions: bool = True     # Opt 3: Numpy array serialization
    use_batch_replay_push: bool = True      # Opt 5: Replay push_batch
    interval_logging_sec: float = 2.0       # Opt 6: Log every N seconds (0 = per-step)
    worker0_verbose_only: bool = True       # Opt 6b: Only rank=0 logs verbose
    queue_maxsize: int = 1000               # Opt 4: Queue backpressure (0 = unbounded)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PerformanceConfig":
        """Create from config dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)
    
    def is_enabled(self, feature: str) -> bool:
        """Check if a specific feature is enabled."""
        if not self.enable_all_optimizations:
            return False
        return getattr(self, feature, False)


@dataclass
class TransitionCounters:
    """
    Transition-level accounting for no-drop verification.
    
    Gate: produced_transitions == consumed_transitions at shutdown.
    Note: qsize() is for monitoring ONLY, never control flow.
    """
    produced_chunks: int = 0
    produced_transitions: int = 0
    consumed_chunks: int = 0
    consumed_transitions: int = 0
    
    def record_produced(self, chunk_count: int) -> None:
        """Call after successful queue.put()."""
        self.produced_chunks += 1
        self.produced_transitions += chunk_count
    
    def record_consumed(self, chunk_count: int) -> None:
        """Call after successful queue.get()."""
        self.consumed_chunks += 1
        self.consumed_transitions += chunk_count
    
    def verify_no_drop(self) -> Tuple[bool, str]:
        """Verify no transitions were dropped."""
        passed = self.produced_transitions == self.consumed_transitions
        msg = (f"produced_transitions={self.produced_transitions}, "
               f"consumed_transitions={self.consumed_transitions}")
        return passed, msg
    
    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class TimingBreakdown:
    """
    Timing instrumentation for bottleneck analysis.
    
    Metrics:
    - env_step_time: Time in simulationStep + state building
    - obs_build_time: Time building observation vectors
    - serialize_time: Time packing transitions for queue
    - learner_update_time: Time in optimizer step
    - replay_sample_time: Time sampling from replay buffer
    - replay_add_time: Time adding to replay buffer
    """
    _times: Dict[str, List[float]] = field(default_factory=dict)
    _starts: Dict[str, float] = field(default_factory=dict)
    
    def start(self, name: str) -> None:
        """Start timing a section."""
        self._starts[name] = time.perf_counter()
    
    def stop(self, name: str) -> float:
        """Stop timing and record."""
        if name not in self._starts:
            return 0.0
        elapsed = time.perf_counter() - self._starts[name]
        if name not in self._times:
            self._times[name] = []
        self._times[name].append(elapsed)
        return elapsed
    
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """Get mean/std/total for each timing category."""
        result = {}
        for name, times in self._times.items():
            if len(times) > 0:
                arr = np.array(times)
                result[name] = {
                    "mean_ms": float(np.mean(arr) * 1000),
                    "std_ms": float(np.std(arr) * 1000),
                    "total_s": float(np.sum(arr)),
                    "count": len(times),
                }
        return result
    
    def summary_str(self) -> str:
        """Human-readable summary."""
        stats = self.get_stats()
        lines = ["Timing Breakdown:"]
        for name, s in stats.items():
            lines.append(f"  {name}: {s['mean_ms']:.2f}ms ± {s['std_ms']:.2f}ms (n={s['count']})")
        return "\n".join(lines)


class IntervalLogger:
    """
    Log at fixed time intervals instead of per-step.
    
    Reduces I/O overhead by >90%.
    """
    def __init__(self, interval_sec: float = 2.0, enabled: bool = True):
        self._interval = interval_sec
        self._enabled = enabled
        self._last_log_time = time.time()
    
    def should_log(self) -> bool:
        """Returns True if enough time has passed since last log."""
        if not self._enabled or self._interval <= 0:
            return True  # Per-step logging
        now = time.time()
        if now - self._last_log_time >= self._interval:
            self._last_log_time = now
            return True
        return False
    
    def reset(self) -> None:
        self._last_log_time = time.time()


@dataclass
class DecisionStepRecord:
    """Single decision step for golden-trace comparison."""
    step_idx: int
    action: int  # Exact match required
    state: List[float]
    reward: float
    done: bool
    gamma: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "action": self.action,
            "state": self.state,
            "reward": self.reward,
            "done": self.done,
            "gamma": self.gamma,
        }


class GoldenTraceRecorder:
    """
    Record decision-step data for invariance testing.
    
    Verification modes:
    - numeric: Actions exact match; floats np.allclose(atol=1e-7, rtol=1e-6)
    - byte: np.ndarray.tobytes() match (same platform/dtype only)
    """
    def __init__(self):
        self._records: List[DecisionStepRecord] = []
        self._step_idx = 0
    
    def record(self, action: int, state: np.ndarray, reward: float, 
               done: bool, gamma: float) -> None:
        """Record a decision step."""
        self._records.append(DecisionStepRecord(
            step_idx=self._step_idx,
            action=int(action),
            state=[float(x) for x in state.flatten().tolist()],
            reward=float(reward),
            done=bool(done),
            gamma=float(gamma),
        ))
        self._step_idx += 1
    
    def save(self, path: Path) -> None:
        """Save trace to JSON file."""
        data = {
            "version": "1.0",
            "total_steps": len(self._records),
            "records": [r.to_dict() for r in self._records],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "GoldenTraceRecorder":
        """Load trace from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        recorder = cls()
        for r in data["records"]:
            recorder._records.append(DecisionStepRecord(**r))
        recorder._step_idx = len(recorder._records)
        return recorder
    
    def compare(self, other: "GoldenTraceRecorder", 
                atol: float = 1e-7, rtol: float = 1e-6) -> Tuple[bool, str]:
        """
        Compare two traces for numeric equivalence.
        
        Actions: exact match required
        Floats: np.allclose with specified tolerances
        """
        if len(self._records) != len(other._records):
            return False, f"Length mismatch: {len(self._records)} vs {len(other._records)}"
        
        for i, (a, b) in enumerate(zip(self._records, other._records)):
            # Actions must match exactly
            if a.action != b.action:
                return False, f"Step {i}: action mismatch {a.action} vs {b.action}"
            
            # States must be close
            if not np.allclose(a.state, b.state, atol=atol, rtol=rtol):
                return False, f"Step {i}: state mismatch"
            
            # Reward must be close
            if not np.isclose(a.reward, b.reward, atol=atol, rtol=rtol):
                return False, f"Step {i}: reward mismatch {a.reward} vs {b.reward}"
            
            # Done must match
            if a.done != b.done:
                return False, f"Step {i}: done mismatch {a.done} vs {b.done}"
            
            # Gamma must be close
            if not np.isclose(a.gamma, b.gamma, atol=atol, rtol=rtol):
                return False, f"Step {i}: gamma mismatch {a.gamma} vs {b.gamma}"
        
        return True, f"All {len(self._records)} steps match (actions exact, floats allclose)"
    
    def compute_hash(self, quantize_precision: float = 1e-7) -> str:
        """
        Compute hash for quick comparison.
        
        Note: hashing uses quantized floats for convenience; 
        the primary acceptance criterion is actions exact match + np.allclose.
        """
        h = hashlib.sha256()
        for r in self._records:
            # Actions: exact
            h.update(str(r.action).encode())
            # Floats: quantized
            for x in r.state:
                h.update(str(round(x / quantize_precision) * quantize_precision).encode())
            h.update(str(round(r.reward / quantize_precision) * quantize_precision).encode())
            h.update(str(int(r.done)).encode())
            h.update(str(round(r.gamma / quantize_precision) * quantize_precision).encode())
        return h.hexdigest()


def compute_throughput(produced_transitions: int, wall_time_sec: float) -> Dict[str, float]:
    """
    Compute throughput metrics.
    
    decision_steps/sec := produced_transitions / wall_time (primary metric)
    """
    if wall_time_sec <= 0:
        return {"decision_steps_per_sec": 0.0, "wall_time_sec": 0.0}
    return {
        "decision_steps_per_sec": produced_transitions / wall_time_sec,
        "wall_time_sec": wall_time_sec,
        "produced_transitions": produced_transitions,
    }


# Sentinel chunk for shutdown signaling
SENTINEL_CHUNK = lambda worker_id: {"type": "END", "worker_id": worker_id, "count": 0}


def is_sentinel(chunk: Dict[str, Any]) -> bool:
    """Check if chunk is a sentinel (shutdown signal)."""
    return chunk.get("type") == "END"
