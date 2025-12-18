import pandas as pd
import matplotlib.pyplot as plt
import os

def load_results(csv_path):
    """Loads simulation data from CSV."""
    if not os.path.exists(csv_path):
        print(f"[ERROR] File not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)

def compute_summary_kpi(df):
    """Computes and prints average KPIs."""
    if df is None: return

    avg_wait = df['Wait_NS'].mean() + df['Wait_EW'].mean()
    avg_queue = df['Queue_NS'].mean() + df['Queue_EW'].mean()
    total_reward = df['Reward'].sum()
    
    print("\n=== SIMULATION PERFORMANCE REPORT ===")
    print(f"Total Steps:            {len(df)}")
    print(f"Average Total Queue:    {avg_queue:.2f} vehicles")
    print(f"Average Total Wait:     {avg_wait:.2f} seconds")
    print(f"Cumulative Reward:      {total_reward:.2f}")
    print("======================================\n")

def plot_performance(df, output_img="outputs/performance.png"):
    """Plots Queue Lengths and Reward over time."""
    if df is None: return

    plt.figure(figsize=(14, 6))
    
    # Subplot 1: Queue Lengths
    plt.subplot(1, 2, 1)
    plt.plot(df['Step'], df['Queue_NS'], label='Queue NS', color='blue', alpha=0.7)
    plt.plot(df['Step'], df['Queue_EW'], label='Queue EW', color='red', alpha=0.7)
    plt.title("Queue Length Evolution")
    plt.xlabel("Simulation Step")
    plt.ylabel("Vehicle Count")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Reward
    plt.subplot(1, 2, 2)
    plt.plot(df['Step'], df['Reward'], label='Reward (Neg Wait)', color='green')
    plt.title("Reward per Step")
    plt.xlabel("Simulation Step")
    plt.ylabel("Reward Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save or Show
    os.makedirs(os.path.dirname(output_img), exist_ok=True)
    plt.savefig(output_img)
    print(f"[INFO] Performance plot saved to {output_img}")
    plt.show()

if __name__ == "__main__":
    # Test execution
    log_file = "outputs/simulation_log.csv"
    data = load_results(log_file)
    compute_summary_kpi(data)
    plot_performance(data)