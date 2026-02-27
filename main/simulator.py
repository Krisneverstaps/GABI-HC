import numpy as np
import pandas as pd
from calculations import calculate_dl

np.random.seed(42)

true_h0 = 70.0
true_om = 0.3

N = 500
zs = np.random.uniform(0.01, 1.0, N)

dl_true = calculate_dl(zs, true_h0, true_om)

sigma = 0.1 * dl_true
dl_obs = dl_true + np.random.normal(0, sigma)

df = pd.DataFrame({
    "z": zs,
    "dl": dl_obs,
    "sigma": sigma
})

df.to_csv("data/gw_events.csv", index=False)

print("Synthetic dataset saved to data/gw_events.csv")