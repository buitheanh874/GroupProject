from __future__ import annotations

import argparse
import os
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import io


def ensure_output_directory(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def select_column(df: pd.DataFrame, candidates: Iterable[str], name: str) -> str:
    for column_name in candidates:
        if column_name in df.columns:
            return column_name
    
    raise ValueError(f"Required column for {name} not found; tried {list(candidates)}")


def compute_summary_by_controller(df: pd.DataFrame, column: str) -> Tuple[List[str], List[float], List[float]]:
    grouped_data = df.groupby("controller")[column].agg(["mean", "std"]).reset_index()
    
    controllers = grouped_data["controller"].astype(str).tolist()
    means = grouped_data["mean"].astype(float).tolist()
    stds = grouped_data["std"].fillna(0.0).astype(float).tolist()
    
    return controllers, means, stds


def plot_bar_chart(
    output_path: str, 
    title: str, 
    y_label: str, 
    controllers: List[str], 
    means: List[float], 
    stds: List[float]
) -> None:
    x_positions = list(range(len(controllers)))
    figure, axis = plt.subplots(figsize=(8, 4))
    
    bars = axis.bar(x_positions, means, yerr=stds, capsize=6)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(controllers)
    axis.set_ylabel(y_label)
    axis.set_title(title)

    for bar, mean_value in zip(bars, means):
        height = bar.get_height()
        axis.text(
            bar.get_x() + bar.get_width() / 2, 
            height, 
            f"{mean_value:.2f}", 
            ha="center", 
            va="bottom", 
            fontsize=9
        )

    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False, help="Path to evaluation CSV file (optional, uses embedded data if not provided)")
    parser.add_argument("--outdir", type=str, default="figures", help="Output directory for figure images")
    args = parser.parse_args()

    output_directory = ensure_output_directory(args.outdir)

    if args.input:
        dataframe = pd.read_csv(args.input)
    else:
        csv_data = """controller,scenario,run_id,total_reward,episode_steps,arrived_vehicles,avg_wait_time,avg_travel_time,avg_stops,avg_queue,max_wait_time,p95_wait_time,throughput
fixed,,0,-271.53858024691357,25,1389,80.61483081353492,268.13894888408925,9.567314614830813,179.87222222222223,441.0,214.5999755859375,55.56
fixed,,1,-448.54320987654313,25,1262,88.95245641838352,295.98335974643425,10.648969889064976,264.81111111111113,486.0,225.0,50.48
fixed,,2,-430.866512345679,25,1342,89.06482861400895,288.54545454545456,10.470938897168406,370.9433333333333,590.0,266.849853515625,53.68
fixed,,3,-604.1574074074075,25,1254,114.97129186602871,344.0135566188198,12.289473684210526,499.96,613.0,344.3499755859375,50.16
fixed,,4,-566.4351851851852,25,1370,109.80510948905109,336.5992700729927,12.375182481751825,468.7388888888889,528.0,310.5499267578125,54.8
fixed,,5,-550.6404320987656,25,1465,117.29556313993174,375.24027303754264,14.563139931740615,442.18,499.0,288.7999267578125,58.6
fixed,,6,-589.1481481481482,25,1328,116.480421686747,354.5745481927711,12.710843373493976,466.24333333333334,620.0,377.0,53.12
fixed,,7,-258.82638888888886,25,1419,78.34319943622269,264.29457364341084,9.569415081042989,164.6511111111111,434.0,216.0999755859375,56.76
fixed,,8,-571.7422839506173,25,1372,111.98542274052478,338.51384839650143,12.495626822157435,466.03,523.0,334.449951171875,54.88
fixed,,9,-591.0216049382716,25,1157,100.3094209161625,304.17977528089887,10.210025929127053,359.58666666666664,555.0,342.39990234375,46.28
max_pressure,,0,-488.63425925925924,25,1378,128.07982583454282,388.9259796806967,14.957910014513788,379.50111111111113,446.0,307.0,55.12
max_pressure,,1,-565.2337962962963,25,1200,123.0675,360.885,13.82,457.69666666666666,554.0,314.0,48.0
max_pressure,,2,-415.8742283950617,25,1209,100.31596360628619,306.37551695616213,10.417700578990901,302.8955555555556,457.0,250.0,48.36
max_pressure,,3,-565.8703703703703,25,1242,123.6231884057971,373.670692431562,13.879227053140097,457.67555555555555,569.0,303.0,49.68
max_pressure,,4,-650.8402777777778,25,1021,113.92752203721841,323.70127326150833,11.059745347698335,402.67333333333335,552.0,290.0,40.84
max_pressure,,5,-623.6126543209876,25,1082,115.21164510166359,331.7513863216266,11.786506469500925,378.8,588.0,276.89990234375,43.28
max_pressure,,6,-430.1543209876542,25,1321,126.59500378501136,363.30431491294473,14.369417108251325,392.15555555555557,458.0,291.0,52.84
max_pressure,,7,-582.3479938271605,25,1156,124.27508650519032,361.71107266435985,13.728373702422145,478.55444444444447,540.0,293.75,46.24
max_pressure,,8,-431.22608024691357,25,1158,97.5293609671848,294.91018998272887,9.62607944732297,320.91333333333336,448.0,254.300048828125,46.32
max_pressure,,9,-468.77932098765433,25,1227,113.32518337408312,329.61450692746536,11.753871230643847,399.5888888888889,564.0,270.0,49.08
rl,,0,-544.5586419753087,25,1238,100.49111470113085,287.2891760904685,9.37156704361874,320.75888888888886,568.0,334.300048828125,49.52
rl,,1,-529.8078703703703,25,1373,102.8353969410051,309.5076474872542,10.694100509832484,459.76444444444445,572.0,319.4000244140625,54.92
rl,,2,-609.2361111111112,25,1364,120.56818181818181,355.38269794721407,12.977272727272727,452.87888888888887,603.0,338.8499755859375,54.56
rl,,3,-274.2276234567902,25,1360,85.52132352941176,262.06617647058823,9.633823529411766,159.41,506.0,241.0,54.4
rl,,4,-434.6288580246913,25,1369,106.60262965668372,301.7034331628926,11.508400292184076,339.0966666666667,529.0,289.5999755859375,54.76
rl,,5,-570.0979938271606,25,1208,96.58443708609272,296.18129139072846,9.06705298013245,342.93333333333334,610.0,309.300048828125,48.32
rl,,6,-584.4266975308643,25,1180,112.87966101694916,302.9957627118644,9.739830508474576,338.5511111111111,601.0,342.0,47.2
rl,,7,-581.3487654320987,25,1098,92.56739526411657,297.92258652094716,9.662112932604735,354.96444444444444,559.0,272.1500244140625,43.92
rl,,8,-693.9567901234568,25,1275,112.76705882352941,305.8666666666667,9.989803921568628,532.2444444444444,581.0,395.2999267578125,51.0
rl,,9,-652.2908950617284,25,1279,112.41907740422205,322.3854573885848,11.885066458170446,413.5466666666667,620.0,307.199951171875,51.16
"""
        dataframe = pd.read_csv(io.StringIO(csv_data))
    
    if "controller" not in dataframe.columns:
        raise ValueError("Input CSV must contain a 'controller' column")

    avg_wait_column = select_column(dataframe, ["avg_wait_time", "avg_wait"], "Average Wait Time")
    
    max_wait_column: Optional[str] = None
    try:
        max_wait_column = select_column(dataframe, ["max_wait_time", "p95_wait_time"], "Max Wait Time")
    except ValueError:
        pass
        
    arrived_column = select_column(dataframe, ["arrived_vehicles", "arrived"], "Arrived Vehicles")

    controllers, avg_means, avg_stds = compute_summary_by_controller(dataframe, avg_wait_column)
    plot_bar_chart(
        output_path=os.path.join(output_directory, "avg_wait.png"),
        title="Average Wait Time by Controller",
        y_label="Average Wait Time (s)",
        controllers=controllers,
        means=avg_means,
        stds=avg_stds,
    )

    if max_wait_column is not None:
        controllers_max, max_means, max_stds = compute_summary_by_controller(dataframe, max_wait_column)
        plot_bar_chart(
            output_path=os.path.join(output_directory, "max_wait.png"),
            title="Tail Wait Time by Controller",
            y_label="Wait Time (s)",
            controllers=controllers_max,
            means=max_means,
            stds=max_stds,
        )

    controllers_arrived, arrived_means, arrived_stds = compute_summary_by_controller(dataframe, arrived_column)
    plot_bar_chart(
        output_path=os.path.join(output_directory, "arrived.png"),
        title="Arrived Vehicles by Controller",
        y_label="Vehicle Count",
        controllers=controllers_arrived,
        means=arrived_means,
        stds=arrived_stds,
    )


if __name__ == "__main__":
    main()