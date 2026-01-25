"""
Create sampled manifests for eval_1_seen_routes from training routes.
Sample 9 routes per demand level (3 per imbalance type).
"""
from pathlib import Path
import random

random.seed(42)

project_root = Path(__file__).resolve().parents[1]
train_final = project_root / "networks" / "variants" / "train_final"
eval_final = project_root / "networks" / "variants" / "eval_final"
eval_final.mkdir(parents=True, exist_ok=True)

# Sample from each level
levels = ["easy", "medium", "hard"]
imbalance_types = ["ns_heavy", "balanced", "ew_heavy"]

all_sampled = []

for level in levels:
    level_dir = train_final / level
    manifest_path = train_final / f"manifest_{level}.txt"
    
    # Read all routes for this level
    routes = manifest_path.read_text(encoding="utf-8").strip().split("\n")
    
    # Group by imbalance type
    by_imbalance = {imb: [] for imb in imbalance_types}
    for route in routes:
        for imb in imbalance_types:
            if imb in route:
                by_imbalance[imb].append(route)
                break
    
    # Sample 3 from each imbalance type
    sampled = []
    for imb in imbalance_types:
        available = by_imbalance[imb]
        if len(available) >= 3:
            sampled.extend(random.sample(available, 3))
        else:
            sampled.extend(available)
    
    print(f"[{level}] Sampled {len(sampled)} routes")
    all_sampled.extend(sampled)

# Write combined manifest for seen routes
seen_manifest = eval_final / "manifest_seen_routes.txt"
seen_manifest.write_text("\n".join(all_sampled), encoding="utf-8")
print(f"\nTotal seen routes: {len(all_sampled)}")
print(f"Saved to: {seen_manifest}")
