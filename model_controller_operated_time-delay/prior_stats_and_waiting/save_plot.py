import os
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt


# def save_plots(dist_pdf, lst_time, lst_N1, lst_N2, lst_ratio):
def save_plots(
    lst_time, lst_N1, lst_N2, lst_ratio, cdf, alpha, beta, dist, model="prior_stats"
):
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    axes[0].plot(lst_time, lst_N1, label="N1")
    axes[0].plot(lst_time, lst_N2, label="N2")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("N1 N2")
    axes[0].set_title("Number of Cars in region 1 and region 2")
    axes[0].legend()
    axes[1].plot(lst_time, lst_ratio, label="load ratio")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("ratio")
    axes[1].set_title("# cars ratio in region 1 over region 2")
    axes[1].legend()

    # plt.legend()
    tr = cdf / (1 - cdf)
    fig.suptitle(f"Parameters: alpha: {alpha}, beta: {beta}\n Traffic Ratio: {tr:5.3}")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_plot = os.path.join(dir_path, "plots/")
    Path(dir_plot).mkdir(parents=True, exist_ok=True)
    file_name = f"{model}_{dist}_alpha_{alpha}_beta_{beta}.png"
    plt.savefig(os.path.join(dir_plot, file_name))


if __name__ == "__main__":
    dir_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(dir_path, "traffic_ratio_vs_scales_weibull.csv"))
    df.plot(x="scales", y="traffic_ratio", kind="scatter")
    plt.show()
