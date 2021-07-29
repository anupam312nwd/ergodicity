import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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


def n_get_critical_point(rv, la):
    n = len(la)
    c = [0] * (n + 1)
    c[0] = -float("inf")
    c[n] = -c[0]
    a = rv.ppf(0.01)  # a=c[i] later
    b = rv.ppf(0.99)
    exp_i_to_inf, exp_i_to_ip1 = [0] * n, [0] * (n - 1)
    for j in tqdm(range(1, n)):
        i = j - 1  # j=1, i=0
        exp_i_to_inf[j] = lambda x: integrate.quad(lambda y: y * rv.pdf(y), x, c[n])[0]
        exp_i_to_ip1[i] = lambda x: integrate.quad(lambda y: y * rv.pdf(y), c[i], x)[0]
        func = lambda x: la[i] * exp_i_to_inf[j](x) - sum(la[j:]) * exp_i_to_ip1[i](x)
        c[j] = optimize.bisect(func, a, b)
        a = c[j]
    return c[1:n]


def traffic_ratio_given_critical_pts(rv, c):
    cp = [0] * (len(c) + 2)
    n = len(cp)
    cp[0], cp[n - 1] = -float("inf"), float("inf")
    cp[1 : n - 1] = c
    traffic_ratio = []
    for j in range(n - 2):
        numerator = rv.cdf(cp[j + 1]) - rv.cdf(cp[j])
        denominator = rv.cdf(cp[j + 2]) - rv.cdf(cp[j + 1])
        traffic_ratio.append(numerator / denominator)
    return traffic_ratio


def get_load_ratio(la):
    load_ratio = []
    for j in range(len(la) - 1):
        load_ratio.append(la[j] / la[j + 1])
    return load_ratio


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


if __name__ == "__main__":
    print("sketch_utils file executed!")
    loc = 20
    scale = 100
    # rv = stats.expon(loc=loc, scale=scale)
    rv = stats.uniform(loc=loc, scale=scale)
    la = [1, 1, 1, 1, 1]
    c = n_get_critical_point(rv=rv, la=la)
    traffic_ratio = traffic_ratio_given_critical_pts(rv, c)
    load_ratio = get_load_ratio(la)
    rounded = lambda x, y: [round(val, y) for val in x]
    print(f"critical points: {rounded(c,3)}")
    print(f"traffic_ratio: {rounded(traffic_ratio, 3)}")
    print(f"load_ratio: {rounded(load_ratio, 3)}")
