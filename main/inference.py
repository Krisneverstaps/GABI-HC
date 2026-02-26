import bilby
import numpy as np
import matplotlib.pyplot as plt
import os

label = 'hubble_inference'
outdir = 'hubble_project_results'
if not os.path.exists(outdir): os.makedirs(outdir)

# Generate Synthetic "Observations" (50 Galaxies)
np.random.seed(42)
true_h0 = 72.0  # The value we want to find
distances = np.linspace(10, 500, 50)  # Mpc
sigma = 1500.0  # Uncertainty in velocity (km/s)
velocities = true_h0 * distances + np.random.normal(0, sigma, 50)

# Model definition
def hubble_law(d, h0):
    return h0 * d

# Priors
priors = dict()
priors['h0'] = bilby.core.prior.Uniform(0, 150, 'h0', latex_label='$H_0$')

# Likelihood
likelihood = bilby.core.likelihood.GaussianLikelihood(
    x=distances, y=velocities, func=hubble_law, sigma=sigma
)

# Sampler (Nested Sampling)
# We use 'unif' as 1D problem
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', 
    nlive=500, sample='unif', outdir=outdir, label=label, clean=True
)


result.plot_corner()
print(f"Inferred H0: {result.posterior['h0'].mean():.2f}")
plt.show()