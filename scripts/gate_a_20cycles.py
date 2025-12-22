from rl.utils import load_yaml_config
from scripts.common import build_env

cfg = load_yaml_config("configs/train_sumo.yaml")
env = build_env(cfg)
env.reset()
for _ in range(20):
    env.step(0)
env.close()
print("OK: 20 cycles done")
