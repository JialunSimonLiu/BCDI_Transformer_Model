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
"""num_epochs = 10
lr = 0.005
input_dim = 3 #d_model
output_dim = 15  # 5 seeds, each with 3 coordinates
nhead = 1
num_layers = 3
dim_feedforward = 128
dropout = 0.1"""
# Crystal parameters
grid_size = 16  # Size of the crystal
radius = grid_size / 2
sigma = 1.5
pad_size = 32

#new
input_dim = 3  # (x, y, amplitude)
output_dim = 3  # Predict (x, y, amplitude)
nhead = 4  # Increased number of heads for better attention
d_model = 128  # Increased model dimension
num_layers = 6  # Increased the number of layers
dim_feedforward = 256  # Increased feedforward dimension for more complex patterns
dropout = 0.1
num_epochs = 200
batch_size = 32
lr = 5e-5
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
"""# Define the Transformer model
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
        return output.permute(1, 0, 2)  # (seq_len, batch_size, output_dim) -> (batch_size, seq_len, output_dim)"""
# Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout, d_model):
        super(TransformerModel, self).__init__()
        # Important! Linear layer to project input_dim to d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Transformer layer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)

        # Linear layer to project d_model back to output_dim
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # Project input to d_model dimension
        src = self.input_proj(src)

        # Pass through transformer
        output = self.transformer(src, src)

        # Project back to output_dim (x, y, intensity)
        output = self.fc_out(output)

        # Ensure x and y are normalized between 0 and 1 and convert back
        output[..., 0] = torch.sigmoid(output[..., 0])  # Normalise x to [0, 1]
        output[..., 1] = torch.sigmoid(output[..., 1])  # Normalise y to [0, 1]

        # Ensure intensity is a value between 0 and 1
        output[..., 2] = torch.sigmoid(output[..., 2])  # Intensity between 0 and 1

        return output
############################################# Loss function #############################################
# see loss_functions.py

############################################# Training #############################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout, d_model).to(device)
# Training setup
optimizer = optim.Adam(model.parameters(), lr=lr)

train_losses = []
eval_losses_per_epoch = []  # Store evaluation losses per epoch
custom_loss = CustomLoss(grid_size=grid_size, radius=radius, sigma=sigma, pad_size=pad_size).to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_train_loss = 0
    for batch_idx, (patterns, seeds) in enumerate(train_loader):
        # Move input data to the same device as the model (GPU or CPU)
        patterns = patterns.to(device)
        seeds = seeds.to(device)

        optimizer.zero_grad()
        output = model(patterns)[:, 0, :]  # Output for the first position

        # Ensure the output and other data are on the same device as the loss function
        loss = custom_loss(output, patterns)
        loss.backward()

        optimizer.step()
        epoch_train_loss += loss.item()

    # Average loss over the entire epoch
    epoch_train_loss /= len(train_loader)
    train_losses.append(epoch_train_loss)

    # Evaluation Phase
    model.eval()
    epoch_eval_loss = 0
    with torch.no_grad():
        eval_losses = []
        for patterns, seeds in test_loader:
            # Move input data to the same device as the model (GPU or CPU)
            patterns = patterns.to(device)
            seeds = seeds.to(device)

            output = model(patterns)[:, 0, :]
            loss = custom_loss(output, patterns)
            eval_losses.append(loss.item())

        mean_eval_loss = np.mean(eval_losses)
        eval_losses_per_epoch.append(mean_eval_loss)

        print(f'Epoch {epoch + 1}/{num_epochs}, Test Loss: {mean_eval_loss}')

# Visualization after final epoch
visualise_final_results(model, test_loader, device, grid_size, pad_size, sigma)

############################################# Prediction #############################################
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
pattern_input_tensor = pattern_input_tensor.to(device)

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
        print(f'Predicted Seed (x, y, phase): ({radius+int(x)}, {radius+int(y)}, {phase:.2f})')

# Plotting the training and evaluation losses
plt.figure(figsize=(12, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(eval_losses_per_epoch, label='Evaluation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss')
plt.legend()
plt.show()
