import heapq
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

from save_plot import save_plots
from utils import (
    get_critical_point,
    get_critical_pt_cdf_expect,
    get_probability,
    shifted_exponential_pdf,
    uniform_pdf,
)


# def generate_simulation(
#     dist_pdf, dist_param, a, b, N1=125, N2=375, total_time=10, alpha=1, beta=1
# ):
def generate_simulation_prior_stats(rv, mu, N1, N2, total_time=10, alpha=1, beta=1):
    lst_N1 = []
    lst_N2 = []
    lst_ratio = []
    lst_time = []
    time = 0
    region_1_entry = 10 * alpha  # initialize for a starting point
    region_2_entry = 10 * beta

    # c, cdf, exp_reg_1, exp_reg_2 = get_critical_pt_cdf_expect(
    #     dist_pdf, alpha=alpha, beta=beta, a=a, b=b, **dist_param
    # )
    _, cdf, exp_reg_1, exp_reg_2 = get_critical_point(rv, alpha=alpha, beta=beta)
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
    return cdf, lst_time, lst_N1, lst_N2, lst_ratio


def generate_simulation_waiting_cars(
    rv, mu, total_time=10, alpha=1, beta=1, num_cars_waiting=50
):
    lst_N1 = []
    lst_N2 = []
    lst_ratio = []
    lst_time = []
    region_1_car_exit_times = []
    region_2_car_exit_times = []
    entry_time = 0
    cars_in_que = []
    _, cdf, _, _ = get_critical_point(rv, alpha=alpha, beta=beta)
    # entry_probability = cdf, 1 - cdf
    partition_num = int(cdf * num_cars_waiting)
    print(f"partition number: {partition_num}")
    # car_estimate = np.random.uniform(lower_bound, upper_bound)
    while entry_time < total_time:
        entry_time += np.random.exponential(1 / mu)
        est_stay = rv.rvs()
        exit_time = entry_time + est_stay
        """entry process"""
        cars_in_que.append(exit_time)
        if len(cars_in_que) == num_cars_waiting:
            cars_in_que.sort()
            cars_to_region_1 = cars_in_que[:partition_num]
            cars_to_region_2 = cars_in_que[partition_num:]
            for ext_t in cars_to_region_1:
                heapq.heappush(region_1_car_exit_times, ext_t)
            for ext_t in cars_to_region_2:
                heapq.heappush(region_2_car_exit_times, ext_t)
            cars_in_que = []
        """exit process"""
        while region_1_car_exit_times and region_1_car_exit_times[0] < entry_time:
            heapq.heappop(region_1_car_exit_times)
        while region_2_car_exit_times and region_2_car_exit_times[0] < entry_time:
            heapq.heappop(region_2_car_exit_times)

        N1 = len(region_1_car_exit_times)
        N2 = len(region_2_car_exit_times)
        if N1 != 0 and N2 != 0:
            lst_N1.append(N1)
            lst_N2.append(N2)
            lst_time.append(entry_time)
            lst_ratio.append(N1 / N2)
    return cdf, lst_time, lst_N1, lst_N2, lst_ratio


if __name__ == "__main__":

    dir_path = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(dir_path, "log_file.log")

    # if not Path(log_file).is_file():
    #     with open(log_file, "w+") as f:
    #         f.write("load_ratio,traffic_ratio\n")

    mu = 20
    loc = 20
    # scale_uniform = 100
    scale_expon = 10
    scale_chi_sq = 10
    # dist_param = {"loc": loc, "scale": scale_uniform}

    # rv, dist = stats.uniform(loc=loc, scale=scale_uniform), "uniform"
    rv, dist = stats.expon(loc=loc, scale=scale_expon), "exponential"
    # rv, dist = stats.chi2(df=5, loc=loc, scale=scale_chi_sq), "chi_sq"

    # dist_pdf = uniform_pdf
    # dist_pdf = shifted_exponential_pdf

    # if dist_pdf.__name__ == "uniform_pdf":
    #     a, b = loc, scale + loc
    # if dist_pdf.__name__ == "shifted_exponential_pdf":
    #     a, b = loc, scale + loc

    model = "expectation_based"
    # model = "waiting_cars"

    # for alpha, beta in tqdm(zip([1, 1, 1, 2, 3], [1, 2, 3, 1, 1])):
    # for alpha, beta in tqdm(zip([1, 1, 1], [1, 2, 3])):
    for alpha, beta in tqdm(zip([1, 1], [1, 2])):
        if model == "expectation_based":
            cdf, timeL, N1_l, N2_l, ratioL = generate_simulation_prior_stats(
                # dist_pdf=dist_pdf,
                # dist_param=dist_param,
                # a=a,
                # b=b,
                rv=rv,
                mu=mu,
                N1=400,
                N2=1000,
                total_time=1000,
                alpha=alpha,
                beta=beta,
            )
        elif model == "waiting_cars":
            cdf, timeL, N1_l, N2_l, ratioL = generate_simulation_waiting_cars(
                rv=rv,
                mu=mu,
                total_time=1000,
                alpha=alpha,
                beta=beta,
                num_cars_waiting=20,
            )
        else:
            raise ValueError(f"model: {model} not defined.")

        save_plots(
            # dist_pdf=dist_pdf,
            lst_time=timeL,
            lst_N1=N1_l,
            lst_N2=N2_l,
            lst_ratio=ratioL,
            cdf=cdf,
            alpha=alpha,
            beta=beta,
            dist=dist,
            model=model,
        )

        print(f"plot saved for alpha = {alpha}, beta = {beta}!")

        # with open(log_file, "a+") as f:
        #     f.write(f"{alpha/beta:.3},{cdf/(1-cdf):.3}\n")
