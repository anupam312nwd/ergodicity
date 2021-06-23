import numpy as np
import random
import matplotlib.pyplot as plt

# initialize N1, N2 : number of vehicles in region 1 and 2
N1, N2 = 125, 375
lst_N1 = []
lst_N2 = []
lst_ratio = []
lst_time = []
alpha = 3
beta = 2
mu = 30
time = 0

while time < 500:
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


plt.plot(lst_time, lst_N1, label="N1")
plt.plot(lst_time, lst_N2, label="N2")
plt.xlabel("time")
plt.ylabel("N1 N2")
plt.legend()
plt.savefig("N1_N2_model_2x2_const_Y.png")
plt.show()

plt.plot(lst_time, lst_ratio)
plt.xlabel("time")
plt.ylabel("ratio")
plt.legend()
plt.savefig("ratio_model_2x2_const_Y.png")
plt.show()
