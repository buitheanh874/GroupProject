import yaml
from sumo_env import SUMOEnv
import analysis
import random
import os

def load_config(config_path="configs/config.yaml"):  
    """Loads the YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def run_simulation():
    # 1. Load Configuration
    try:
        # You can also pass a custom path here if needed
        cfg = load_config("configs/config.yaml") 
        print(f"[INFO] Configuration loaded from configs/config.yaml")
    except Exception as e:
        print(f"[ERROR] Failed to load config: {e}")
        return

    # 2. Initialize Environment
    # scenario='high' will use the route file defined in yaml
    env = SUMOEnv(cfg, scenario='high') 
    
    state = env.reset()
    done = False
    
    # Get run limit from config
    max_steps = cfg['simulation']['max_steps']
    total_reward = 0
    
    print(f"[INFO] Starting simulation for {max_steps} steps...")

    try:
        # 3. Main Simulation Loop
        while not done and env.step_counter < max_steps:
            
            # --- AGENT LOGIC ---
            # Random action between 0.1 and 0.9
            action = random.uniform(0.1, 0.9)
            # -------------------
            
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            
            # Print log every 100 steps
            if env.step_counter % 100 == 0:
                print(f"Step {env.step_counter}/{max_steps} | Action: {action:.2f} | Reward: {reward:.1f}")
                
            state = next_state
            
        print(f"[INFO] Simulation finished. Total Reward: {total_reward:.2f}")

    except KeyboardInterrupt:
        print("\n[WARN] Simulation interrupted by user.")
        
    finally:
        # 4. Cleanup
        env.save_logs()
        env.close()
        
        # Trigger Analysis
        log_path = cfg['simulation']['log_file']
        if os.path.exists(log_path):
            print("[INFO] Running analysis...")
            df = analysis.load_results(log_path)
            analysis.compute_summary_kpi(df)

if __name__ == "__main__":
    run_simulation()