"""Generate manifest_mix.txt combining routes from all demand levels."""
from __future__ import annotations
import random
from pathlib import Path

def main():
    project_root = Path(__file__).resolve().parents[1]
    base_dir = project_root / "networks/variants/train_final"
    
    # Load all manifests
    easy_routes = [line.strip() for line in 
                   (base_dir / "manifest_easy.txt").read_text().splitlines() if line.strip()]
    medium_routes = [line.strip() for line in 
                     (base_dir / "manifest_medium.txt").read_text().splitlines() if line.strip()]
    hard_routes = [line.strip() for line in 
                   (base_dir / "manifest_hard.txt").read_text().splitlines() if line.strip()]
    
    print(f"Easy: {len(easy_routes)}, Medium: {len(medium_routes)}, Hard: {len(hard_routes)}")
    
    # Mix ratio: 60% easy, 30% medium, 10% hard
    # For 100 routes in mix pool:
    random.seed(42)
    
    mix_routes = []
    mix_routes.extend(random.sample(easy_routes, 60))   # 60% easy
    mix_routes.extend(random.sample(medium_routes, 30)) # 30% medium  
    mix_routes.extend(random.sample(hard_routes, 10))   # 10% hard
    
    # Shuffle the mix
    random.shuffle(mix_routes)
    
    # Write manifest
    output = base_dir / "manifest_mix.txt"
    output.write_text("\n".join(mix_routes) + "\n")
    
    print(f"\nGenerated {output}: {len(mix_routes)} routes")
    print(f"  - Easy: 60, Medium: 30, Hard: 10")

if __name__ == "__main__":
    main()
