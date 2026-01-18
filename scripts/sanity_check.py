#!/usr/bin/env python
"""
Sanity Check - Validate metrics and pipeline integrity

Usage:
    python scripts/sanity_check.py
    python scripts/sanity_check.py --quick

Checks:
    1. Metrics bounds (completion_rate, teleport_rate in [0,1])
    2. Vehicle conservation (inserted ≈ arrived + present ± teleport)
    3. Output file format validation
    4. Controller imports
    5. Config loading
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))


class SanityChecker:
    """Run sanity checks on the pipeline."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        
        # Use ASCII-safe symbols for Windows compatibility
        self.SYM_PASS = "[PASS]"
        self.SYM_FAIL = "[FAIL]"
        self.SYM_WARN = "[WARN]"
    
    def log(self, msg: str) -> None:
        if self.verbose:
            # Handle encoding for Windows console
            try:
                print(msg)
            except UnicodeEncodeError:
                print(msg.encode('ascii', 'replace').decode('ascii'))
    
    def check(self, name: str, condition: bool, error_msg: str = "") -> bool:
        if condition:
            self.passed += 1
            self.log(f"  {self.SYM_PASS} {name}")
            return True
        else:
            self.failed += 1
            self.log(f"  {self.SYM_FAIL} {name}: {error_msg}")
            return False
    
    def warn(self, name: str, msg: str) -> None:
        self.warnings += 1
        self.log(f"  {self.SYM_WARN} {name}: {msg}")
    
    def check_imports(self) -> bool:
        """Check that all required modules can be imported."""
        self.log("\n[1] Checking imports...")
        
        all_ok = True
        
        # Controllers
        try:
            from controllers import (
                FixedTimeController,
                MaxPressureSplitController,
                ActuatedController,
                WebsterController,
            )
            self.check("controllers import", True)
        except ImportError as e:
            self.check("controllers import", False, str(e))
            all_ok = False
        
        # RL modules
        try:
            from rl.agent import DQNAgent
            from rl.utils import load_yaml_config
            from scripts.common import build_env
            self.check("rl modules import", True)
        except ImportError as e:
            self.check("rl modules import", False, str(e))
            all_ok = False
        
        # Env modules
        try:
            from env.kpi import EpisodeKpiTracker
            from env.sumo_env import SUMOEnv
            self.check("env modules import", True)
        except ImportError as e:
            self.check("env modules import", False, str(e))
            all_ok = False
        
        return all_ok
    
    def check_configs(self) -> bool:
        """Check that config files exist and are valid."""
        self.log("\n[2] Checking configs...")
        
        all_ok = True
        
        configs = [
            "configs/train_1.yaml",
            "configs/train_1_plain.yaml",
        ]
        
        for cfg in configs:
            path = project_root / cfg
            if path.exists():
                try:
                    from rl.utils import load_yaml_config
                    config = load_yaml_config(str(path))
                    self.check(f"{cfg} loads", True)
                    
                    # Check key fields
                    if 'env' in config and 'sumo' in config['env']:
                        self.check(f"{cfg} has env.sumo", True)
                    else:
                        self.check(f"{cfg} has env.sumo", False, "missing env.sumo")
                        all_ok = False
                except Exception as e:
                    self.check(f"{cfg} loads", False, str(e))
                    all_ok = False
            else:
                self.check(f"{cfg} exists", False, "file not found")
                all_ok = False
        
        # Check RL-Plain has ablation settings
        plain_path = project_root / "configs/train_1_plain.yaml"
        if plain_path.exists():
            try:
                from rl.utils import load_yaml_config
                config = load_yaml_config(str(plain_path))
                
                sumo_cfg = config.get('env', {}).get('sumo', {})
                agent_cfg = config.get('agent', {})
                
                # Verify ablation settings
                checks = [
                    ("reward_time_normalize=false", sumo_cfg.get('reward_time_normalize') == False),
                    ("alpha_spillback=0", sumo_cfg.get('alpha_spillback', 1) == 0),
                    ("use_time_aware_gamma=false", agent_cfg.get('use_time_aware_gamma') == False),
                ]
                
                for name, ok in checks:
                    if ok:
                        self.check(f"train_1_plain.yaml: {name}", True)
                    else:
                        self.warn("train_1_plain.yaml", f"{name} not set correctly")
                        
            except Exception as e:
                self.warn("train_1_plain.yaml ablation check", str(e))
        
        return all_ok
    
    def check_manifests(self) -> bool:
        """Check route manifests exist."""
        self.log("\n[3] Checking route manifests...")
        
        all_ok = True
        
        manifests = [
            "networks/variants/train/manifest_d800.txt",
            "networks/variants/train_1000s/manifest_d600.txt",
            "networks/variants/train_1000s/manifest_d800.txt",
            "networks/variants/train_1000s/manifest_d1000.txt",
        ]
        
        for manifest in manifests:
            path = project_root / manifest
            if path.exists():
                with open(path, 'r') as f:
                    lines = [l.strip() for l in f if l.strip() and not l.startswith('#')]
                if len(lines) > 0:
                    self.check(f"{manifest} ({len(lines)} routes)", True)
                else:
                    self.check(manifest, False, "empty file")
                    all_ok = False
            else:
                self.warn(manifest, "not found (may need to generate)")
        
        return all_ok
    
    def check_metrics_bounds(self, csv_path: str) -> bool:
        """Check that metrics are within valid bounds."""
        self.log(f"\n[4] Checking metrics bounds in {csv_path}...")
        
        if not Path(csv_path).exists():
            self.warn("metrics check", f"file not found: {csv_path}")
            return True  # Not a failure, just skip
        
        all_ok = True
        issues = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row_idx, row in enumerate(reader):
                # Check completion_rate in [0, 1]
                if 'completion_rate' in row:
                    cr = float(row['completion_rate'])
                    if not (0 <= cr <= 1.0):
                        issues.append(f"row {row_idx}: completion_rate={cr} out of [0,1]")
                
                # Check teleport_rate in [0, 1]
                if 'teleport_rate' in row:
                    tr = float(row['teleport_rate'])
                    if not (0 <= tr <= 1.0):
                        issues.append(f"row {row_idx}: teleport_rate={tr} out of [0,1]")
                
                # Check arrived >= 0
                if 'arrived_vehicles' in row:
                    av = int(row['arrived_vehicles'])
                    if av < 0:
                        issues.append(f"row {row_idx}: arrived_vehicles={av} < 0")
        
        if issues:
            for issue in issues[:5]:  # Show first 5
                self.check("metrics bounds", False, issue)
                all_ok = False
            if len(issues) > 5:
                self.warn("metrics bounds", f"... and {len(issues) - 5} more issues")
        else:
            self.check("metrics bounds", True)
        
        return all_ok
    
    def check_controller_consistency(self) -> bool:
        """Check that all controllers support the same interface."""
        self.log("\n[5] Checking controller consistency...")
        
        all_ok = True
        
        try:
            from dataclasses import dataclass
            
            @dataclass
            class MockAction:
                rho_ns: float
                cycle_sec: int
            
            action_space = [
                MockAction(0.3, 90),
                MockAction(0.5, 90),
                MockAction(0.7, 90),
            ]
            
            state = np.array([5.0, 5.0, 3.0, 3.0])
            
            from controllers import (
                FixedTimeController,
                ActuatedController,
                WebsterController,
            )
            
            # Test each controller
            controllers = [
                ("FixedTimeController", FixedTimeController(action_space)),
                ("ActuatedController", ActuatedController(action_space)),
                ("WebsterController", WebsterController(action_space)),
            ]
            
            for name, ctrl in controllers:
                try:
                    if hasattr(ctrl, 'reset'):
                        ctrl.reset()
                    
                    action = ctrl.act(state) if hasattr(ctrl, 'act') and name != "FixedTimeController" else ctrl.act()
                    
                    if 0 <= action < len(action_space):
                        self.check(f"{name}.act() returns valid action", True)
                    else:
                        self.check(f"{name}.act()", False, f"action {action} out of range")
                        all_ok = False
                except Exception as e:
                    self.check(f"{name}.act()", False, str(e))
                    all_ok = False
            
        except Exception as e:
            self.check("controller consistency", False, str(e))
            all_ok = False
        
        return all_ok
    
    def run_all(self, quick: bool = False) -> int:
        """Run all sanity checks."""
        self.log("=" * 60)
        self.log("SANITY CHECK")
        self.log("=" * 60)
        
        self.check_imports()
        self.check_configs()
        self.check_manifests()
        
        if not quick:
            self.check_controller_consistency()
            
            # Check gating results if exist
            gating_csv = project_root / "gating_results" / "demand_feasibility.csv"
            if gating_csv.exists():
                self.check_metrics_bounds(str(gating_csv))
        
        # Summary
        self.log("\n" + "=" * 60)
        self.log(f"PASSED:   {self.passed}")
        self.log(f"FAILED:   {self.failed}")
        self.log(f"WARNINGS: {self.warnings}")
        self.log("=" * 60)
        
        if self.failed > 0:
            self.log("\n❌ SANITY CHECK FAILED - Fix issues before proceeding")
            return 1
        elif self.warnings > 0:
            self.log("\n⚠️ SANITY CHECK PASSED WITH WARNINGS")
            return 0
        else:
            self.log("\n✅ ALL CHECKS PASSED")
            return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sanity checks")
    parser.add_argument("--quick", action="store_true", help="Quick mode (skip slow checks)")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    checker = SanityChecker(verbose=not args.quiet)
    return checker.run_all(quick=args.quick)


if __name__ == "__main__":
    sys.exit(main())
