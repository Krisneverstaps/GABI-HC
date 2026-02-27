import torch
import numpy as np
from train_emulator import DLEmulator

# Load metadata for normalization
input_mean = torch.from_numpy(np.load("input_mean.npy")).float()
input_std = torch.from_numpy(np.load("input_std.npy")).float()
target_mean = torch.from_numpy(np.load("target_mean.npy")).float()
target_std = torch.from_numpy(np.load("target_std.npy")).float()

# Initialize Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DLEmulator()
model.load_state_dict(torch.load("dl_emulator.pth", map_location=device))
model.to(device).eval()

def calculate_dl(zs, h0, omega_m):
    """
    High-performance wrapper for Bilby.
    """
    # Prepare inputs as a batch (Vectorized)
    # We do this in torch to keep it ready for the GPU,as my windows is not able to load CUDA for now
    zs_t = torch.as_tensor(zs).float()
    h0_t = torch.full_like(zs_t, h0)
    om_t = torch.full_like(zs_t, omega_m)
    
    inputs = torch.stack([zs_t, h0_t, om_t], dim=1).to(device)

    # Normalize and Predict
    inputs_norm = (inputs - input_mean.to(device)) / input_std.to(device)
    
    with torch.no_grad():
        outputs_norm = model(inputs_norm)

    # De-normalize
    outputs = outputs_norm * target_std.to(device) + target_mean.to(device)
    
    return outputs.cpu().numpy().flatten()