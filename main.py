from sumo_env import SUMOEnv
import analysis  # Assuming analysis.py exists from previous step
import random
import os

def run_simulation_scenario(scenario_type='low', controller_type='random'):
    """
    Runs simulation using BI.net.xml and BI_*.rou.xml
    """
    
    # --- PATHS UPDATED FOR YOUR FILES ---
    net_file = "networks/BI.net.xml"
    
    if scenario_type == 'low':
        # Using your provided Low Demand file
        route_file = "networks/BI_50_test.rou.xml" 
    else:
        # Using your provided High Demand file
        route_file = "networks/BI_150_test.rou.xml"
        
    log_file = f"outputs/log_{scenario_type}_{controller_type}.csv"

    # Initialize Environment
    # Check if files exist
    if not os.path.exists(net_file) or not os.path.exists(route_file):
        print(f"[ERROR] Files not found. Please put BI.net.xml and .rou.xml inside 'networks/' folder.")
        return

    env = SUMOEnv(net_file, route_file, use_gui=True, log_file=log_file)
    
    print(f"--- STARTING SIMULATION: Scenario={scenario_type.upper()} | Controller={controller_type.upper()} ---")
    
    state = env.reset()
    done = False
    total_reward = 0
    
    try:
        while not done:
            # --- CONTROLLER LOGIC ---
            if controller_type == 'random':
                # Random split for NS Green (0.2 to 0.8)
                action = random.uniform(0.2, 0.8)
            elif controller_type == 'fixed':
                # Fixed 50-50 split
                action = 0.5
            
            # Step
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            
            print(f"Step {env.step_counter:4d} | Action: {action:.2f} | Q_NS: {int(state[0]):3d} | Q_EW: {int(state[1]):3d} | Reward: {reward:.1f}")
            state = next_state
            
    except KeyboardInterrupt:
        print("\n[WARN] Simulation stopped by user.")
        
    finally:
        env.save_logs()
        env.close()
        
        # Simple Analysis Trigger
        print("\nGenerating Analysis...")
        df = analysis.load_results(log_file)
        analysis.compute_summary_kpi(df)
        # analysis.plot_performance(df) # Uncomment if you want plots popup

if __name__ == "__main__":
    # Example: Run High Demand (BI_150) with Random Controller
    run_simulation_scenario(scenario_type='high', controller_type='random')