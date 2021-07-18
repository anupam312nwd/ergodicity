import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize
from scipy.optimize.zeros import bisect


def uniform_pdf(x, lower_bd=20, upper_bd=120):
    if lower_bd <= x <= upper_bd:
        return 1 / (upper_bd - lower_bd)
    else:
        return 0


def find_critical_point(alpha=2, beta=3, lower_bd=-5, upper_bd=300):
    func = (
        lambda x: beta * integrate.quad(uniform_pdf, -float("inf"), x)[0]
        - alpha * integrate.quad(uniform_pdf, x, float("inf"))[0]
    )
    return optimize.bisect(
        func,
        lower_bd,
        upper_bd,
    )


def get_expectation_region_1_2(upper=80):
    cp = find_critical_point(upper_bd=80)
    expectation_region_1 = integrate.quad(
        lambda x: x * uniform_pdf(x), -float("inf"), cp
    )[0]
    expectation_region_2 = integrate.quad(
        lambda x: x * uniform_pdf(x), cp, float("inf")
    )[0]
    return expectation_region_1, expectation_region_2


def get_probability(*args):
    args = np.array(args)
    return args / np.sum(args)


def generate_simulation(N1=125, N2=375, total_time=10, alpha=1, beta=1):
    lst_N1 = []
    lst_N2 = []
    lst_ratio = []
    lst_time = []
    mu = 30
    time = 0
    lower_bound = 20
    upper_bound = 120

    exp_reg_1, exp_reg_2 = get_expectation_region_1_2()
    car_estimate = np.random.uniform(lower_bound, upper_bound)

    while time < total_time:
        time += np.random.exponential(1 / mu)
        X = random.choices([1, 2], [alpha / (alpha + beta), beta / (alpha + beta)])[0]

        if X == 1:
            N1 += 1
        else:
            N2 += 1

        Y = random.choices([1, 2], get_probability(N1 / exp_reg_1, N2 / exp_reg_2))[0]
        if Y == 1:
            N1 -= 1
        else:
            N2 -= 1

        lst_N1.append(N1)
        lst_N2.append(N2)
        lst_time.append(time)
        lst_ratio.append(N1 / N2)
    return lst_time, lst_N1, lst_N2, lst_ratio


if __name__ == "__main__":
    lst_time, lst_N1, lst_N2, lst_ratio = generate_simulation(
        N1=200, N2=200, total_time=200, alpha=1, beta=1
    )

    fig, axes = plt.subplots(1, 2)
    axes[0].plot(lst_time, lst_N1, label="N1")
    axes[0].plot(lst_time, lst_N2, label="N2")
    axes[0].set_xlabel("time")
    axes[0].set_ylabel("N1 N2")
    axes[0].legend()
    axes[1].plot(lst_time, lst_ratio)
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("ratio")
    plt.legend()
    # plt.savefig("./uniform_dist_plots/N1_N2_model.png")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_plot = os.path.join(dir_path, "plots/")
    Path(dir_plot).mkdir(parents=True, exist_ok=True)
    plt.savefig(os.path.join(dir_plot, "N1_N2_model.png"))
    plt.show()
