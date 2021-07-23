import math
import time
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import integrate, optimize, stats
from scipy.optimize.zeros import bisect

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


if __name__ == "__main__":
    loc = 20
    scale = 10

    rv = stats.uniform(loc=loc, scale=scale)
    dist_pdf = uniform_pdf

    # rv = stats.expon(loc=loc, scale=scale)
    # dist_pdf = shifted_exponential_pdf

    if dist_pdf.__name__ == "uniform_pdf":
        a, b = loc, scale + loc
    if dist_pdf.__name__ == "shifted_exponential_pdf":
        a, b = loc, scale + loc

    start_time = time.time()
    print(get_critical_point(rv))
    first_time = time.time()
    print(f"took {first_time - start_time} seconds")
    print(get_critical_pt_cdf_expect(dist_pdf, loc=loc, scale=scale, a=a, b=b))
    second_time = time.time()
    print(f"took {second_time - first_time} seconds")
