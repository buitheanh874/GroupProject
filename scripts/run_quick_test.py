import subprocess
import sys
from scripts.repo_root import find_repo_root

repo_root = find_repo_root(__file__)

def main():
    print("="*80)
    print("QUICK TEST PIPELINE (5 episodes each)")
    print("="*80)
    
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "run_full_pipeline.py"),
        "--episodes", "5",
        "--eval-runs", "2",
    ]
    
    result = subprocess.run(cmd, cwd=repo_root)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
