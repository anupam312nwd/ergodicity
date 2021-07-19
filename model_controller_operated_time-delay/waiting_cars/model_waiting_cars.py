import heapq
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize
from scipy.optimize.zeros import bisect
from tqdm import tqdm

"""only exit_time is needed for simulation."""
# class car:
#     def __init__(self, entry_time, est_stay):
#         self.entry_time = entry_time
#         self.est_stay = est_stay

#     @property
#     def exit_time(self):
#         return self.entry_time + self.est_stay


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


def generate_simulation(total_time=10, alpha=1, beta=1, num_cars_waiting=20):
    lst_N1 = []
    lst_N2 = []
    lst_ratio = []
    lst_traffic_ratio = []
    lst_time = []
    region_1_car_exit_times = []
    region_2_car_exit_times = []
    mu = 30
    entry_time = 0
    lower_bound = 20
    upper_bound = 120
    num_cars_waiting = num_cars_waiting
    cars_in_que = []
    cp, cdf, exp_reg_1, exp_reg_2 = get_critical_point_cdf_expectation(
        alpha=alpha, beta=beta, lower_bd=lower_bound, upper_bd=upper_bound
    )
    entry_probability = cdf, 1 - cdf
    partition_num = int(cdf * num_cars_waiting)
    print(f"partition number: {partition_num}")
    # car_estimate = np.random.uniform(lower_bound, upper_bound)
    while entry_time < total_time:
        entry_time += np.random.exponential(1 / mu)
        est_stay = np.random.uniform(low=lower_bound, high=upper_bound)
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
        # lst_traffic_ratio.append(region_1_entry / region_2_entry)
    return cdf, lst_time, lst_N1, lst_N2, lst_ratio


def save_plots(lst_time, lst_N1, lst_N2, lst_ratio, num_cars_waiting):
    fig, axes = plt.subplots(1, 2, figsize=(17, 10))
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
    plt.legend()
    # plt.savefig("./uniform_dist_plots/N1_N2_model.png")
    dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_plot = os.path.join(dir_path, "plots_after_normalization/")
    Path(dir_plot).mkdir(parents=True, exist_ok=True)
    plt.savefig(
        os.path.join(
            dir_plot,
            f"waiting_cars_{num_cars_waiting}_model_alpha_{alpha}_beta_{beta}.png",
        )
    )


if __name__ == "__main__":
    num_cars_waiting = 50
    list_alpha_beta = [
        (1, 1),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
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
            # lst_traffic_ratio,
        ) = generate_simulation(
            total_time=1000, alpha=alpha, beta=beta, num_cars_waiting=num_cars_waiting
        )
        save_plots(
            lst_time=lst_time,
            lst_N1=lst_N1,
            lst_N2=lst_N2,
            lst_ratio=lst_ratio,
            # lst_traffic_ratio=lst_traffic_ratio,
            num_cars_waiting=num_cars_waiting,
        )

        print(f"plot saved for alpha = {alpha}, beta = {beta}!")

        with open(log_file, "a+") as f:
            f.write(f"{alpha/beta:.3},{cdf/(1-cdf):.3}\n")
