import torch
import numpy as np
from train_emulator import DLEmulator

input_mean = np.load("input_mean.npy")
input_std = np.load("input_std.npy")
target_mean = np.load("target_mean.npy")
target_std = np.load("target_std.npy")

device = torch.device("cpu")

model = DLEmulator()
model.load_state_dict(torch.load("dl_emulator.pth"))
model.eval()

def calculate_dl(zs, h0, omega_m):
    inputs = np.stack([
        zs,
        np.full_like(zs, h0),
        np.full_like(zs, omega_m)
    ], axis=1)

    # Normalize inputs
    inputs_norm = (inputs - input_mean) / input_std

    inputs_tensor = torch.tensor(inputs_norm, dtype=torch.float32)

    with torch.no_grad():
        outputs_norm = model(inputs_tensor).numpy().flatten()

    # De-normalize outputs
    outputs = outputs_norm * target_std + target_mean

    return outputs