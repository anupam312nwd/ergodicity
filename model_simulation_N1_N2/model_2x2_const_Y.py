import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

# initialize N1, N2 : number of vehicles in region 1 and 2


def get_time_series_N1_N2_ratio_Y(N1=125, N2=375, total_time=500):
    # initialize N1, N2 : number of vehicles in region 1 and 2
    # N1, N2 = N1, N2
    lst_N1 = []
    lst_N2 = []
    lst_ratio = []
    lst_time = []
    alpha = 3
    beta = 2
    mu = 30
    time = 0

    while time < total_time:
        time += np.random.exponential(1 / mu)
        Z = alpha / N1 + beta / N2
        X = random.choices([1, 2], [alpha / (N1 * Z), beta / (N2 * Z)])[0]

        if X == 1:
            N1 += 1
        else:
            N2 += 1

        Z = N1 + N2
        # Y = random.choices([1, 2], (N1 / Z, N2 / Z))[0]
        Y = random.choices([1, 2], (0.5, 0.5))[0]
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

    lst_time, lst_N1, lst_N2, lst_ratio = get_time_series_N1_N2_ratio_Y()

    plt.plot(lst_time, lst_N1, label="N1")
    plt.plot(lst_time, lst_N2, label="N2")
    plt.xlabel("time")
    plt.ylabel("N1 N2")
    plt.legend()
    plt.savefig("N1_N2_model_2x2_const_Y.png")
    plt.show()

    ratio_df = pd.Series(lst_ratio, index=lst_time)
    plt.plot(lst_time, lst_ratio)
    plt.xlabel("time")
    plt.ylabel("ratio")
    plt.legend()
    plt.savefig("ratio_model_2x2_const_Y.png")
    plt.show()

    ratio_df.plot(style="k--")
    ratio_df_moving_avg = ratio_df.rolling(60).mean()
    pd.rolling_mean(ratio_df, 60).plot(style="k")
    pd.rolling_std(ratio_df, 30).plot(style="b")
    plt.savefig("ratio_model_2x2_const_Y_w_mean_std.png")
    plt.show()
