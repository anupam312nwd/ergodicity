import math
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate, optimize, special, stats
from scipy.optimize.zeros import bisect
from tqdm import tqdm

from traffic_simulation import shifted_exponential_pdf, uniform_pdf


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


def plot_traffic_vs_load_ratio():
    load_ratio = []
    traffic_ratio_chi2 = []
    traffic_ratio_uniform = []
    traffic_ratio_exponential = []

    lst_rv = []
    loc = 20
    lst_rv.append(stats.expon(loc=loc, scale=50))
    lst_rv.append(stats.chi2(df=5, loc=loc, scale=10))
    lst_rv.append(stats.uniform(loc=loc, scale=100))

    for alpha, beta in tqdm(
        zip(
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7, 9, 11, 2, 1, 1, 1, 1, 1],
        )
    ):
        load_ratio.append(alpha / beta)

        for rv, rv_name in zip(lst_rv, ["exponential", "chi2", "uniform"]):
            _, cdf_c, _, _ = get_critical_point(rv, alpha=alpha, beta=beta)
            tr = cdf_c / (1 - cdf_c)
            if rv_name == "exponential":
                traffic_ratio_exponential.append(tr)
            elif rv_name == "chi2":
                traffic_ratio_chi2.append(tr)
            elif rv_name == "uniform":
                traffic_ratio_uniform.append(tr)
            else:
                raise ValueError("rv not found!")

    dct = {
        "load_ratio": load_ratio,
        "traffic_chi2": traffic_ratio_chi2,
        "traffic_uniform": traffic_ratio_uniform,
        "traffic_exponential": traffic_ratio_exponential,
    }

    dir_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.DataFrame.from_dict(dct)
    df.to_csv(os.path.join(dir_path, "traffic_vs_load_ratio.csv"), index=False)

    plt.figure(figsize=(12, 10))
    plt.plot(load_ratio, traffic_ratio_exponential, label="exponential")
    plt.plot(load_ratio, traffic_ratio_chi2, label="chi2")
    plt.plot(load_ratio, traffic_ratio_uniform, label="uniform")
    plt.xlabel("load_ratio")
    plt.ylabel("traffic_ratio")
    plt.legend()
    plt.savefig(os.path.join(dir_path, "traffic_vs_load_ratio.png"))


def plot_traffic_vs_load_from_file():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(dir_path, "traffic_vs_load_ratio.csv"))
    plt.figure(figsize=(12, 10))
    plt.scatter(x="load_ratio", y="traffic_chi2", data=df, label="chi2")
    plt.scatter(x="load_ratio", y="traffic_uniform", data=df, label="uniform")
    plt.scatter(x="load_ratio", y="traffic_exponential", data=df, label="exponential")
    plt.scatter(df["load_ratio"], df["load_ratio"], label="y=x")
    plt.xlabel("load_ratio")
    plt.ylabel("traffic_ratio")
    plt.legend()
    plt.savefig(os.path.join(dir_path, "traffic_vs_load_ratio.png"))
    plt.show()


if __name__ == "__main__":

    plot_traffic_vs_load_from_file()
    loc = 20
    scale = 100

    rv = stats.uniform(loc=loc, scale=scale)
    dist_pdf = uniform_pdf

    # rv = stats.expon(loc=loc, scale=scale)
    # dist_pdf = shifted_exponential_pdf

    rv = stats.chi2(df=5, loc=loc, scale=scale)

    if dist_pdf.__name__ == "uniform_pdf":
        a, b = loc, scale + loc
    if dist_pdf.__name__ == "shifted_exponential_pdf":
        a, b = loc, scale + loc

    # for alpha, beta in zip([1, 1, 1, 2, 3], [1, 2, 3, 1, 1]):
    #     start_time = time.time()
    #     print(f"for alpha={alpha}, beta={beta}:")
    #     print(get_critical_point(rv, alpha=alpha, beta=beta))
    #     first_time = time.time()
    #     print(f"took {first_time - start_time} seconds")

    #     print(
    #         get_critical_pt_cdf_expect(
    #             dist_pdf, loc=loc, scale=scale, alpha=alpha, beta=beta, a=a, b=b
    #         )
    #     )
    #     second_time = time.time()
    #     print(f"took {second_time - first_time} seconds")
