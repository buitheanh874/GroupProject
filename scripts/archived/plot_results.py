import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

csv_content = """controller,scenario,run_id,total_reward,episode_steps,arrived_vehicles,avg_wait_time,avg_travel_time,avg_stops,avg_queue,max_wait_time,p95_wait_time,throughput
fixed,,0,-594.7,25,1278,113.5,345.6,12.6,475.8,617.0,348.2,51.12
fixed,,1,-565.8,25,1408,118.9,352.2,13.1,442.9,516.0,340.7,56.32
fixed,,2,-594.4,25,1328,119.5,354.4,13.6,487.4,589.0,313.7,53.12
fixed,,3,-678.1,25,1268,106.4,307.6,11.1,573.0,617.0,317.3,50.72
fixed,,4,-653.2,25,1380,113.1,365.9,13.6,593.2,525.0,334.1,55.20
fixed,,5,-601.0,25,1307,103.6,328.2,12.0,469.0,562.0,278.7,52.28
fixed,,6,-680.4,25,1243,94.9,289.0,9.7,571.2,577.0,278.9,49.72
fixed,,7,-536.4,25,1370,106.0,351.8,13.4,445.9,563.0,269.5,54.80
fixed,,8,-681.7,25,1391,115.1,369.4,13.6,601.2,580.0,323.0,55.64
fixed,,9,-682.5,25,1300,102.8,303.6,10.7,519.6,609.0,356.1,52.00
max_pressure,,0,-584.5,25,1210,137.2,385.0,14.9,432.1,535.0,349.0,48.40
max_pressure,,1,-596.9,25,1241,138.1,382.4,14.4,481.8,490.0,322.0,49.64
max_pressure,,2,-583.3,25,1186,134.0,378.9,14.3,470.2,521.0,337.3,47.44
max_pressure,,3,-596.6,25,1209,136.4,394.3,15.5,518.5,485.0,323.6,48.36
max_pressure,,4,-590.6,25,1242,133.7,371.5,14.0,447.5,493.0,327.9,49.68
max_pressure,,5,-581.7,25,1272,135.7,377.3,14.4,434.9,511.0,330.4,50.88
max_pressure,,6,-630.7,25,1134,135.7,375.2,14.5,486.2,612.0,324.7,45.36
max_pressure,,7,-600.4,25,1237,128.7,379.7,14.4,478.9,558.0,304.0,49.48
max_pressure,,8,-575.9,25,1183,127.8,372.9,14.0,466.9,614.0,316.0,47.32
max_pressure,,9,-571.3,25,1174,131.2,379.6,14.6,452.4,547.0,327.0,46.96
rl,,0,-766.7,25,1161,104.5,286.3,9.3,560.2,567.0,330.0,46.44
rl,,1,-697.7,25,1274,122.9,342.0,12.7,517.4,525.0,355.3,50.96
rl,,2,-712.7,25,1298,119.8,318.1,11.3,544.3,565.0,352.2,51.92
rl,,3,-572.5,25,1463,121.7,365.5,13.7,425.6,562.0,326.9,58.52
rl,,4,-594.6,25,1413,117.3,346.7,12.6,436.3,614.0,323.4,56.52
rl,,5,-597.6,25,1377,125.8,349.8,12.8,446.0,539.0,370.0,55.08
rl,,6,-789.2,25,1164,110.6,295.2,9.3,560.4,596.0,350.7,46.56
rl,,7,-569.5,25,1420,128.2,352.2,13.1,410.0,587.0,361.0,56.80
rl,,8,-557.3,25,1483,121.3,341.0,12.3,414.49,571.0,357.9,59.32
rl,,9,-594.7,25,1418,99.8,326.1,11.8,428.1,619.0,310.2,56.72
"""

data_frame = pd.read_csv(io.StringIO(csv_content))

controller_names = {
    'fixed': 'Fixed Time',
    'max_pressure': 'Max Pressure',
    'rl': 'RL Agent (AI)'
}
data_frame['Controller'] = data_frame['controller'].map(controller_names)

sns.set(style="whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.barplot(
    data=data_frame,
    x="Controller",
    y="avg_wait_time",
    ax=axes[0],
    palette="viridis",
    capsize=0.1
)
axes[0].set_title("Average Waiting Time (Lower is Better)", fontsize=14, fontweight='bold')
axes[0].set_ylabel("Seconds")
axes[0].set_xlabel("")

sns.barplot(
    data=data_frame,
    x="Controller",
    y="arrived_vehicles",
    ax=axes[1],
    palette="magma",
    capsize=0.1
)
axes[1].set_title("Arrived Vehicles (Higher is Better)", fontsize=14, fontweight='bold')
axes[1].set_ylabel("Vehicle Count")
axes[1].set_xlabel("")

sns.barplot(
    data=data_frame,
    x="Controller",
    y="total_reward",
    ax=axes[2],
    palette="coolwarm",
    capsize=0.1
)
axes[2].set_title("Total Reward (Higher is Better)", fontsize=14, fontweight='bold')
axes[2].set_ylabel("Reward")
axes[2].set_xlabel("")

plt.tight_layout()
plt.savefig("comparison_chart.png")
print("Chart generated successfully! Please open 'comparison_chart.png' to view.")
plt.show()