import os
import pandas as pd
from pathlib import Path

from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# dir_path = os.path.dirname(os.path.abspath(''))
dir_path = globals()["_dh"][0]

df = pd.read_csv(os.path.join(dir_path, "traffic_vs_load_ratio.csv"))
df_sorted = df.sort_values(by=["load_ratio"])

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[1].plot(
    df_sorted["load_ratio"], df_sorted["traffic_exponential"], label="exponential"
)
ax[1].plot(df_sorted["load_ratio"], df_sorted["traffic_chi2"], label="chi2")
ax[1].plot(df_sorted["load_ratio"], df_sorted["traffic_uniform"], label="uniform")
ax[1].plot(df_sorted["load_ratio"], df_sorted["load_ratio"], label="y=x")
ax[1].set_ylabel("traffic_ratio")
ax[1].set_xlabel("load_ratio")
ax[1].legend()

loc = 20
rv_expon = stats.expon(loc=loc, scale=50)
rv_chi2 = stats.chi2(df=5, loc=loc, scale=10)
rv_uniform = stats.uniform(loc=20, scale=100)
lst_rv = []
lst_rv.append(rv_expon)
lst_rv.append(rv_chi2)
lst_rv.append(rv_uniform)
for rv, rv_name in zip(lst_rv, ["exponential", "chi2", "uniform"]):
    x = np.linspace(rv.ppf(0.01), rv.ppf(0.99), 100)
    ax[0].plot(x, rv.pdf(x), lw=2, alpha=0.6, label=rv_name)
ax[0].set_ylabel("pdf")
ax[0].set_xlabel("x")
ax[0].legend()
plt.savefig("pdf_and_traffic_vs_load_ratio.png")
plt.show()
