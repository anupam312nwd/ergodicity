import math
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats
from scipy.optimize.zeros import bisect
from tqdm import tqdm


def uniform_pdf(x, loc=20, scale=100):
    lower_bd = loc
    upper_bd = loc + scale
    if lower_bd <= x <= upper_bd:
        return 1 / (upper_bd - lower_bd)
    else:
        return 0


def shifted_exponential_pdf(x, loc, scale):
    lower_bd = loc
    mu = 1 / scale
    if lower_bd <= x:
        return mu * math.exp(-mu * (x - lower_bd))
    else:
        return 0


def get_critical_point(rv, alpha=1, beta=1):
    a = rv.ppf(0.01)
    b = rv.ppf(0.99)
    expect_1 = lambda x: integrate.quad(lambda y: y * rv.pdf(y), -float("inf"), x)[0]
    expect_2 = lambda x: integrate.quad(lambda y: y * rv.pdf(y), x, float("inf"))[0]
    func = lambda x: beta * expect_1(x) - alpha * expect_2(x)
    c = optimize.bisect(func, a, b)
    cdf_c = rv.cdf(c)
    return c, cdf_c, expect_1(c) / cdf_c, expect_2(c) / (1 - cdf_c)


def get_critical_pt_cdf_expect(dist_pdf, a, b, alpha=1, beta=1, **kwargs):
    pdf = lambda x: dist_pdf(x, **kwargs)
    cdf = lambda x: integrate.quad(pdf, a, x)[0]
    expect_1 = lambda x: integrate.quad(lambda y: y * pdf(y), -float("inf"), x)[0]
    expect_2 = lambda x: integrate.quad(lambda y: y * pdf(y), x, float("inf"))[0]
    func = lambda x: beta * expect_1(x) - alpha * expect_2(x)
    c = optimize.bisect(func, a, b)
    cdf_c = cdf(c)
    return c, cdf_c, expect_1(c) / cdf_c, expect_2(c) / (1 - cdf_c)


def get_probability(*args):
    args = np.array(args)
    return args / np.sum(args)


# def generate_simulation(
#     dist_pdf, dist_param, a, b, N1=125, N2=375, total_time=10, alpha=1, beta=1
# ):
def generate_simulation(rv, N1=125, N2=375, total_time=10, alpha=1, beta=1):
    lst_N1 = []
    lst_N2 = []
    lst_ratio = []
    lst_traffic_ratio = []
    lst_time = []

    mu = 1
    time = 0
    region_1_entry = 10 * alpha  # initialize for a starting point
    region_2_entry = 10 * beta

    # c, cdf, exp_reg_1, exp_reg_2 = get_critical_pt_cdf_expect(
    #     dist_pdf, alpha=alpha, beta=beta, a=a, b=b, **dist_param
    # )
    c, cdf, exp_reg_1, exp_reg_2 = get_critical_point(rv, alpha=alpha, beta=beta)
    entry_probability = cdf, 1 - cdf
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


# def save_plots(dist_pdf, lst_time, lst_N1, lst_N2, lst_ratio):
def save_plots(lst_time, lst_N1, lst_N2, lst_ratio):
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
    plt.savefig(
        os.path.join(
            # dir_plot, f"prior_stat_{dist_pdf.__name__}_alpha_{alpha}_beta_{beta}.png"
            dir_plot,
            f"prior_stat_chi_square_alpha_{alpha}_beta_{beta}.png",
        )
    )


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(dir_path, "log_file.log")

    if not Path(log_file).is_file():
        with open(log_file, "w+") as f:
            f.write("load_ratio,traffic_ratio\n")

    loc = 20
    scale = 10
    dist_param = {"loc": loc, "scale": scale}

    # rv = stats.uniform(loc=loc, scale=scale)
    # dist_pdf = uniform_pdf

    rv = stats.expon(loc=loc, scale=scale)
    dist_pdf = shifted_exponential_pdf

    # rv = stats.chi2(df=5, loc=loc, scale=scale)

    if dist_pdf.__name__ == "uniform_pdf":
        a, b = loc, scale + loc
    if dist_pdf.__name__ == "shifted_exponential_pdf":
        a, b = loc, scale + loc

    # for alpha, beta in tqdm(zip([1, 1, 1, 2, 3], [1, 2, 3, 1, 1])):
    for alpha, beta in tqdm(zip([1, 1], [1, 2])):
        (
            cdf,
            lst_time,
            lst_N1,
            lst_N2,
            lst_ratio,
            lst_traffic_ratio,
        ) = generate_simulation(
            # dist_pdf=dist_pdf,
            # dist_param=dist_param,
            # a=a,
            # b=b,
            rv=rv,
            N1=100,
            N2=2900,
            total_time=30000,
            alpha=alpha,
            beta=beta,
        )
        save_plots(
            # dist_pdf=dist_pdf,
            lst_time=lst_time,
            lst_N1=lst_N1,
            lst_N2=lst_N2,
            lst_ratio=lst_ratio,
        )

        print(f"plot saved for alpha = {alpha}, beta = {beta}!")

        with open(log_file, "a+") as f:
            f.write(f"{alpha/beta:.3},{cdf/(1-cdf):.3}\n")
