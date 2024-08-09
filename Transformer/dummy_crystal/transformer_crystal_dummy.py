import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import math
from loss_functions import *

############################################# Parameters #############################################
# Model parameters
num_epochs = 5
input_dim = 3 #d_model
output_dim = 15  # 5 seeds, each with 3 coordinates
nhead = 1
num_layers = 3
dim_feedforward = 128
dropout = 0.1
# Crystal parameters
grid_size = 16  # Size of the crystal
radius = grid_size / 2
sigma = 1.5
pad_size = 32

############################################# Data processing #############################################
# Load the HDF5 file
file_path = 'data_train.h5'
with h5py.File(file_path, 'r') as hdf:
    # List all groups
    groups = list(hdf.keys())
    seeds = []
    patterns = []
    for group in groups:
        data = hdf.get(group)
        seed_values = data['seeds'][:]
        pattern_values = data['diffraction_patterns'][:]
        seeds.append(seed_values)
        patterns.append(pattern_values)

seeds = np.array(seeds)
patterns = np.array(patterns)
# Flatten seeds for each group
seeds = seeds.reshape(seeds.shape[0], -1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(patterns, seeds, test_size=0.2, random_state=42)

# Preparing the dataset for DataLoader
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

############################################# Transformer Model #############################################
# Define the Transformer model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerModel, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(input_dim, output_dim)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # (batch_size, seq_len, input_dim) -> (seq_len, batch_size, input_dim)
        output = self.transformer(src, src)
        output = self.fc_out(output)
        return output.permute(1, 0, 2)  # (seq_len, batch_size, output_dim) -> (batch_size, seq_len, output_dim)

model = TransformerModel(input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout)

############################################# Loss function #############################################
def custom_loss(output, targets, grid_size, radius, sigma = sigma, pad_size = pad_size):
    device = output.device
    batch_size = output.size(0)

    # Reshape output and targets to (batch_size, num_seeds, 3)
    output = output.view(batch_size, -1, 3)
    targets = targets.view(batch_size, -1, 3)
    # Ensure x and y in output are integers and within the supporting circle
    output[:, :, 0] = torch.clamp(output[:, :, 0].round(), min=0, max=grid_size - 1)
    output[:, :, 1] = torch.clamp(output[:, :, 1].round(), min=0, max=grid_size - 1)
    # Ensure x and y in output are integers and within the supporting circle
    circle_center = grid_size //2
    output[:, :, 0:2] = output[:, :, 0:2].clone().round().int()
    output = support(output, circle_center, radius)
    #print(output)
    # Create real-space crystals from seeds
    #predicted_real_space = torch.zeros(batch_size, pad_size, pad_size, requires_grad=True).to(device)
    predicted_real_space_list = []
    for batch_idx in range(batch_size):
        seeds_pred = output[batch_idx].clone().detach().cpu().numpy()
        #print("predicted seeds", radius + seeds_pred)
        padded_voronoi_pred = create_circular_voronoi_diagram(seeds_pred, grid_size, pad_size)
        phase_pred, _, _ = assign_phase_values(padded_voronoi_pred, seeds_pred, grid_size, pad_size)
        phase_pred_tensor = torch.tensor(phase_pred, dtype=torch.float32, device=device)
        temp_real_space = create_density_function(1.0, phase_pred_tensor).to(device)
        #predicted_real_space[batch_idx] = temp_real_space
        predicted_real_space_list.append(temp_real_space.unsqueeze(0))
    predicted_real_space = torch.cat(predicted_real_space_list, dim=0)

    """plt.imshow(phase_pred, cmap='viridis')
    plt.colorbar()
    plt.title('Phase Predictions')
    plt.show()"""

    # Fourier transform to get diffraction patterns
    predicted_real_space = torch.fft.fftshift(predicted_real_space, dim=(-2, -1))
    predicted_dp = torch.fft.fft2(predicted_real_space)
    predicted_dp = torch.fft.fftshift(predicted_dp, dim=(-2, -1)).abs()
    target_dp = create_diffraction_pattern(targets, grid_size, sigma)

    """plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_dp[1].cpu().numpy(), cmap='hot', origin='lower')
    plt.title('Predicted Diffraction Pattern')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(target_dp[1].cpu().numpy(), cmap='hot', origin='lower')
    plt.title('Target Diffraction Pattern')
    plt.colorbar()
    plt.show()"""

    target_dp = target_dp.requires_grad_(True)
    # Calculate chi-square loss
    loss = calculate_chi_square(predicted_dp, target_dp)
    #print(loss)
    return loss

############################################# Training #############################################
# Training setup
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses = []
eval_losses = []
prediction_losses = []
# Training loop
model.train()
for epoch in range(num_epochs):
    epoch_train_loss = 0
    for batch_idx, (patterns, seeds) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(patterns)[:, 0, :]  # Output for the first position
        loss = custom_loss(output, patterns, grid_size, radius)
        loss.backward()
        """for name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Gradient for {name}: {param.grad}')
            else:
                print(f'No gradient for {name}')"""
        optimizer.step()
        epoch_train_loss += loss.item()

    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # Output seed information
    with torch.no_grad():
        for patterns, seeds in train_loader:
            output = model(patterns)[:, 0, :]
            rounded_output = output.clone()
            rounded_output[:, 0::3] = torch.clamp(rounded_output[:, 0::3].round(), min=0)  # Rounding x
            rounded_output[:, 1::3] = torch.clamp(rounded_output[:, 1::3].round(), min=0)  # Rounding y
            for i in range(rounded_output.size(0)):
                seed_info = rounded_output[i].cpu().numpy().reshape(-1, 3)
                for seed in seed_info:
                    x, y, phase = seed
                    #print(f'Epoch {epoch + 1}, Seed (x, y, phase): ({int(x)}, {int(y)}, {phase:.2f})')
            break  # Only show one batch for brevity

    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_train_loss}')

# Evaluation
model.eval()
with torch.no_grad():
    for patterns, seeds in test_loader:
        output = model(patterns)[:, 0, :]
        loss = custom_loss(output, patterns, grid_size, radius)
        eval_losses.append(loss.item())
    mean_eval_loss = np.mean(eval_losses)
    print(f'Test Loss: {mean_eval_loss}')

# Prediction
pattern_input = np.array([
    [16, 17, 1.0], [16, 14, 0.6691], [15, 11, 0.2762], [17, 24, 0.1463], [20, 17, 0.1337],
    [15, 8, 0.1328], [14, 5, 0.1104], [12, 14, 0.1055], [20, 21, 0.0875], [12, 10, 0.0817],
    [18, 30, 0.0776], [14, 1, 0.0747], [12, 24, 0.0641], [19, 5, 0.0604], [20, 8, 0.0593],
    [13, 27, 0.0561], [11, 19, 0.056], [9, 24, 0.0496], [21, 13, 0.0474], [21, 28, 0.0453],
    [11, 4, 0.0448], [23, 8, 0.0448], [11, 7, 0.0435], [21, 25, 0.0428], [8, 18, 0.0415],
    [9, 20, 0.0412], [23, 21, 0.0388], [23, 12, 0.0372], [28, 15, 0.0359], [22, 1, 0.0358],
    [4, 15, 0.0346], [9, 10, 0.0339], [7, 14, 0.0287], [25, 28, 0.0243], [3, 10, 0.0238],
    [2, 6, 0.0234], [8, 8, 0.0218], [26, 1, 0.0193], [5, 6, 0.0177], [27, 26, 0.0173], [5, 28, 0.0168]
])

pattern_input_tensor = torch.tensor(pattern_input, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

model.eval()
with torch.no_grad():
    output = model(pattern_input_tensor)[:, 0, :]
    rounded_output = output.clone()
    rounded_output[:, 0::3] = torch.clamp(rounded_output[:, 0::3].round(), min=0)  # Rounding x
    rounded_output[:, 1::3] = torch.clamp(rounded_output[:, 1::3].round(), min=0)  # Rounding y
    seed_info = rounded_output[0].cpu().numpy().reshape(-1, 3)
    print("Predicted Seed Information:")
    for seed in seed_info:
        x, y, phase = seed
        print(f'Predicted Seed (x, y, phase): ({int(x)}, {int(y)}, {phase:.2f})')

# Plotting the training and evaluation losses
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(range(len(eval_losses)), eval_losses, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.show()
