import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


dir_path = os.path.dirname(os.path.abspath(__file__))
# log_file = os.path.join(dir_path, "log_file.log")
log_file = os.path.join(dir_path, "log_file_after_normalization.log")

if Path(log_file).is_file():
    df = pd.read_csv(log_file)
    df["traffic_to_load"] = df["traffic_ratio"] / df["load_ratio"]
    # df.plot.scatter(x="load_ratio", y="traffic_ratio", figsize=(15, 15))
    df.plot.scatter(x="load_ratio", y="traffic_to_load", figsize=(15, 15))
    x = df["load_ratio"].to_numpy()
    plt.savefig(os.path.join(dir_path, "traffic_to_load_ratio.png"))

    # values = [0.2, 0.5, 1.0, 2.0]
    # for val in values:
    #     dfn = df.loc[df["load_ratio"] < val]
    #     dfn.plot.scatter(x="load_ratio", y="traffic_ratio", figsize=(15, 15))
    #     x = dfn["load_ratio"].to_numpy()
    #     plt.plot(x, x)
    #     plt.savefig(
    #         os.path.join(dir_path, f"load_less_than_{val:.3}_traffic_ratio.png")
    #     )

sns.heatmap()