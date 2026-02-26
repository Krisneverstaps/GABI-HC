import cupy as cp
import numpy as np
import pandas as pd
import time

def calculate_dl_gpu(zs, h0, omega_m=0.3):
    """GPU-accelerated Luminosity Distance calculation."""
    c = 299792.458 
    # Create integration grid for each z in zs
    # We use a 2D grid to vectorize the integration
    z_grid = cp.linspace(0, zs, 100) 
    
    def ez(z):
        return cp.sqrt(omega_m * (1 + z)**3 + (1 - omega_m))
    
    integrand = 1.0 / ez(z_grid)
    integral = cp.trapz(integrand, z_grid, axis=0)
    return (c * (1 + zs) / h0) * integral

# Generate Synthetic GW Dataset
np.random.seed(42)
true_h0 = 70.0
true_om = 0.3
zs = np.random.uniform(0.01, 1.0, 100) # Redshifts up to z=1

# Move to GPU for calculation
zs_gpu = cp.array(zs)
start = time.time()
dl_true = calculate_dl_gpu(zs_gpu, true_h0, true_om).get() # Move back to CPU
gpu_time = time.time() - start

# Add Gaussian noise (typical 10% uncertainty for GW distances)
sigma = 0.1 * dl_true
dl_obs = dl_true + np.random.normal(0, sigma)

# Save dataset
df = pd.DataFrame({'z': zs, 'dl': dl_obs, 'sigma': sigma})
df.to_csv('data/gw_events.csv', index=False)
print(f"Dataset generated. GPU Calc Time: {gpu_time:.5f}s")