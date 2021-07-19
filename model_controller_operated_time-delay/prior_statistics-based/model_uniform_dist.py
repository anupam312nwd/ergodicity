import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize
from scipy.optimize.zeros import bisect
from tqdm import tqdm


def uniform_pdf(x, lower_bd=20, upper_bd=120):
    if lower_bd <= x <= upper_bd:
        return 1 / (upper_bd - lower_bd)
    else:
        return 0


def get_critical_point_cdf_expectation(alpha=2, beta=3, lower_bd=20, upper_bd=120):
    pdf = lambda x: uniform_pdf(x, lower_bd=lower_bd, upper_bd=upper_bd)
    cdf = lambda x: integrate.quad(pdf, lower_bd, x)[0]
    expectation_region_1 = lambda x: integrate.quad(
        lambda y: y * pdf(y), -float("inf"), x
    )[0]
    expectation_region_2 = lambda x: integrate.quad(
        lambda y: y * pdf(y), x, float("inf")
    )[0]
    func = lambda x: beta * expectation_region_1(x) - alpha * expectation_region_2(x)
    cp = optimize.bisect(
        func,
        lower_bd - 5,
        upper_bd + 5,
    )
    return (
        cp,
        cdf(cp),
        expectation_region_1(cp) / cdf(cp),
        expectation_region_2(cp) / (1 - cdf(cp)),
    )


def get_probability(*args):
    args = np.array(args)
    return args / np.sum(args)


def generate_simulation(N1=125, N2=375, total_time=10, alpha=1, beta=1):
    lst_N1 = []
    lst_N2 = []
    lst_ratio = []
    lst_traffic_ratio = []
    lst_time = []
    mu = 30
    time = 0
    lower_bound = 20
    upper_bound = 120
    region_1_entry = 10 * alpha  # initialize for a starting point
    region_2_entry = 10 * beta
    cp, cdf, exp_reg_1, exp_reg_2 = get_critical_point_cdf_expectation(
        alpha=alpha, beta=beta, lower_bd=lower_bound, upper_bd=upper_bound
    )
    entry_probability = cdf, 1 - cdf
    car_estimate = np.random.uniform(lower_bound, upper_bound)
    while time < total_time:
        time += np.random.exponential(1 / mu)

        """entry process"""
        X = random.choices([1, 2], entry_probability)[0]
        if X == 1:
            N1 += 1
            region_1_entry += 1
        else:
            N2 += 1
            region_2_entry += 1

        """exit process"""
        exit_probability = get_probability(N1 / exp_reg_1, N2 / exp_reg_2)
        Y = random.choices([1, 2], exit_probability)[0]
        if Y == 1:
            N1 -= 1
        else:
            N2 -= 1
        lst_N1.append(N1)
        lst_N2.append(N2)
        lst_time.append(time)
        lst_ratio.append(N1 / N2)
        lst_traffic_ratio.append(region_1_entry / region_2_entry)
    return cdf, lst_time, lst_N1, lst_N2, lst_ratio, lst_traffic_ratio


def save_plots(lst_time, lst_N1, lst_N2, lst_ratio, lst_traffic_ratio):
    fig, axes = plt.subplots(2, 2, figsize=(17, 17))
    axes[0, 0].plot(lst_time, lst_N1, label="N1")
    axes[0, 0].plot(lst_time, lst_N2, label="N2")
    axes[0, 0].set_xlabel("time")
    axes[0, 0].set_ylabel("N1 N2")
    axes[0, 0].set_title("Number of Cars in region 1 and region 2")
    axes[0, 0].legend()
    axes[0, 1].plot(lst_time, lst_ratio, label="load ratio")
    axes[0, 1].set_xlabel("time")
    axes[0, 1].set_ylabel("ratio")
    axes[0, 1].set_title("# cars ratio in region 1 over region 2")
    axes[0, 1].legend()
    axes[1, 0].plot(lst_time, lst_traffic_ratio, label="traffic ratio")
    axes[1, 0].set_title("traffic ratio in region 1 over region 2")
    axes[1, 0].legend()
    plt.legend()
    # plt.savefig("./uniform_dist_plots/N1_N2_model.png")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_plot = os.path.join(dir_path, "plots_after_normalization/")
    Path(dir_plot).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(dir_plot, f"prior_stat_model_alpha_{alpha}_beta_{beta}.png")
    )


if __name__ == "__main__":
    alpha = 1
    beta = 1
    list_alpha_beta = [
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (2, 3),
        (3, 2),
        (1, 4),
        (1, 5),
        (1, 6),
        (1, 7),
        (2, 5),
        (2, 7),
        (4, 3),
        (4, 1),
        (5, 1),
        (1, 8),
        (1, 9),
        (1, 10),
        (1, 11),
        (1, 12),
        (6, 1),
        (7, 1),
        (8, 1),
        (1, 13),
        (1, 14),
        (1, 15),
        (1, 18),
        (1, 20),
        (1, 25),
    ]

    dir_path = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(dir_path, "log_file_after_normalization.log")

    if not Path(log_file).is_file():
        with open(log_file, "w+") as f:
            f.write("load_ratio,traffic_ratio\n")

    for alpha, beta in tqdm(list_alpha_beta):
        (
            cdf,
            lst_time,
            lst_N1,
            lst_N2,
            lst_ratio,
            lst_traffic_ratio,
        ) = generate_simulation(
            N1=1500, N2=1500, total_time=1000, alpha=alpha, beta=beta
        )
        save_plots(
            lst_time=lst_time,
            lst_N1=lst_N1,
            lst_N2=lst_N2,
            lst_ratio=lst_ratio,
            lst_traffic_ratio=lst_traffic_ratio,
        )

        print(f"plot saved for alpha = {alpha}, beta = {beta}!")

        with open(log_file, "a+") as f:
            f.write(f"{alpha/beta:.3},{cdf/(1-cdf):.3}\n")
