import numpy as np
from scipy.integrate import quad

c = 299792.458  # km/s

def E(z, omega_m):
    return np.sqrt(omega_m * (1 + z)**3 + (1 - omega_m))

def luminosity_distance(z, h0, omega_m):
    integral, _ = quad(lambda zp: 1.0 / E(zp, omega_m), 0, z)
    return (c * (1 + z) / h0) * integral

def calculate_dl(zs, h0, omega_m):
    return np.array([luminosity_distance(z, h0, omega_m) for z in zs])