import random

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


class car:
    def __init__(self, est_stay, entry_time):
        self.est_stay = est_stay
        self.entry_time = entry_time
