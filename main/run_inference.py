import bilby
import pandas as pd
import numpy as np
from calculations import calculate_dl as analytic
from for_emulator import calculate_dl as emulator

zs_test = np.linspace(0.01, 2.0, 50)
h0_test = 70.0
om_test = 0.3

dl_true = analytic(zs_test, h0_test, om_test)
dl_em = emulator(zs_test, h0_test, om_test)

frac_err = np.abs(dl_true - dl_em) / dl_true

print("Mean fractional error:", frac_err.mean())
print("Max fractional error:", frac_err.max())


# Choose backend
USE_EMULATOR = True

if USE_EMULATOR:
    from for_emulator import calculate_dl
else:
    from calculations import calculate_dl

data = pd.read_csv("data/gw_events.csv")

zs = data["z"].values
dl_obs = data["dl"].values
sigmas = data["sigma"].values

def model_func(z, h0, omega_m):
    return calculate_dl(z, h0, omega_m)

likelihood = bilby.core.likelihood.GaussianLikelihood(
    x=zs,
    y=dl_obs,
    func=model_func,
    sigma=sigmas
)

priors = dict(
    h0=bilby.core.prior.Uniform(50, 100, name="h0"),
    omega_m=bilby.core.prior.Uniform(0.1, 0.5, name="omega_m")
)

result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="dynesty",
    nlive=2000,
    outdir="results",
    label="h0_inference"
)

result.plot_corner()