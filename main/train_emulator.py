import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from calculations import calculate_dl

device = torch.device("cpu")

class DLEmulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.net(x)

# Generate training data
N = 10000
h0_samples = np.random.uniform(50, 90, N)
om_samples = np.random.uniform(0.1, 0.5, N)
z_samples = np.random.uniform(0.0, 1.5, N)

inputs_np = np.stack([z_samples, h0_samples, om_samples], axis=1)

targets_np = np.array([
    calculate_dl([z], h0, om)[0]
    for z, h0, om in zip(z_samples, h0_samples, om_samples)
])

# Normalize inputs 
input_mean = inputs_np.mean(axis=0)
input_std = inputs_np.std(axis=0)

inputs_np_norm = (inputs_np - input_mean) / input_std

# Normalize targets 
target_mean = targets_np.mean()
target_std = targets_np.std()

targets_np_norm = (targets_np - target_mean) / target_std

# Convert to tensors
inputs = torch.tensor(inputs_np_norm, dtype=torch.float32)
targets = torch.tensor(targets_np_norm, dtype=torch.float32).unsqueeze(1)

model = DLEmulator().to(device)
optimizer = optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.MSELoss()

epochs = 500

for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), "dl_emulator.pth")
np.save("input_mean.npy", input_mean)
np.save("input_std.npy", input_std)
np.save("target_mean.npy", target_mean)
np.save("target_std.npy", target_std)
print("Model saved as dl_emulator.pth")