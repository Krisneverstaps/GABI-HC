import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from calculations import calculate_dl

# EMULATOR CLASS
class DLEmulator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Normalizes the layer to prevent "exploding gradients"
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

# SETTINGS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N = 100000  

# GENERATE TRAINING DATA
print(f"Generating {N} data points on CPU...")
h0_samples = np.random.uniform(50, 100, N)
om_samples = np.random.uniform(0.1, 0.5, N)
z_samples = np.random.uniform(0.0, 1.5, N)

inputs_np = np.stack([z_samples, h0_samples, om_samples], axis=1)

# will take a while
targets_np = np.array([
    calculate_dl([z], h0, om)[0]
    for z, h0, om in zip(z_samples, h0_samples, om_samples)
])

# NORMALIZATION (Metadata)
input_mean = inputs_np.mean(axis=0)
input_std = inputs_np.std(axis=0)
target_mean = targets_np.mean()
target_std = targets_np.std()

inputs_norm = (inputs_np - input_mean) / input_std
targets_norm = (targets_np - target_mean) / target_std

# Convert to tensors
inputs_t = torch.tensor(inputs_norm, dtype=torch.float32)
targets_t = torch.tensor(targets_norm, dtype=torch.float32).unsqueeze(1)

# tRAINING SETUP
model = DLEmulator().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

# TRAINING LOOP
epochs = 500 # With 100k points and BatchNorm, 500 is a good amount, can test more later
batch_size = 1024 

print(f"Starting training on {device}...")
for epoch in range(epochs):
    model.train()
    # Shuffling data for better learning
    permutation = torch.randperm(inputs_t.size(0))
    
    epoch_loss = 0
    for i in range(0, inputs_t.size(0), batch_size):
        optimizer.zero_grad()
        indices = permutation[i:i+batch_size]
        batch_x, batch_y = inputs_t[indices].to(device), targets_t[indices].to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_loss = epoch_loss / (N // batch_size)
    scheduler.step(avg_loss)

    if epoch % 25 == 0:
        print(f"Epoch {epoch:03d} | Loss: {avg_loss:.8f} | LR: {optimizer.param_groups[0]['lr']}")

# SAVE OUTPUTS
torch.save(model.state_dict(), "dl_emulator.pth")
np.save("input_mean.npy", input_mean)
np.save("input_std.npy", input_std)
np.save("target_mean.npy", target_mean)
np.save("target_std.npy", target_std)

print("\nSuccess! Training complete and model saved.")