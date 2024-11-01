# Jialun Liu UCL/LCN 24.10.2024
# This supervised transformer model has the input: diffraction pattern (x, y, amplitude), the model Output: predicted crystal in the form of (x, y, amplitude, phase)
# Test full information from a diffraction pattern

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader, random_split
from transformer_model import *
from data_loader import *
from functions import *

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

############################################ Hyperparameters ############################################
img_size = 32
N_seed = 6
num_epochs = 100
batch_size = 32
learning_rate = 1e-4

input_dim = 3  # Diffraction pattern (x, y, amplitude)
output_dim = 4  # Predict (x, y, amplitude, phase)
nhead = 4
d_model = 128
num_layers = 6
dim_feedforward = 256
dropout = 0.1

############################################ Dataset loading ############################################
# Loading H5 files for diffraction pattern and seeds
dp_file_path = 'training_data/train_simulated_full_dp.h5'
seeds_file_path = 'training_data/train_simulated_seeds.h5'

# Initialise DataLoader, see data_loader.py
train_loader, eval_loader = create_data_loader(batch_size=batch_size, dp_file_path=dp_file_path, seeds_file_path=seeds_file_path)

############################################ Functions ############################################
# See functions.py

############################################ Transformer Model ############################################
# See transformer_model.py

############################################ Train and Evaluation ############################################
# Training and evaluation function
def train_complex_model(model, optimizer, criterion, train_loader, eval_loader, num_epochs):
    train_losses = []
    eval_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Average training loss
        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Evaluation phase
        model.eval()
        eval_loss = 0.0
        with torch.no_grad():
            for inputs, targets in eval_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                #print(outputs)
                loss = criterion(outputs, targets)
                eval_loss += loss.item()

        # Average evaluation loss
        avg_eval_loss = eval_loss / len(eval_loader)
        eval_losses.append(avg_eval_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Eval Loss: {avg_eval_loss:.4f}")

    # Plot the losses
    plt.figure(figsize=(10, 6))
    plt.plot(np.log10(train_losses), label='Training Loss')
    plt.plot(np.log10(eval_losses), label='Evaluation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Evaluation Loss')
    plt.legend()
    plt.show()

    model.eval()
    return model

# Plot function to compare diffraction patterns and reconstructed crystals
def plot_comparison(model, test_diffraction_pattern, true_seed, img_size):
    # Predict the crystal from the diffraction pattern
    predicted_crystal = model(test_diffraction_pattern.to(device)).detach().cpu()

    # Convert predicted crystal to amplitude and phase
    predicted_amplitude = predicted_crystal[..., 2].view(img_size, img_size)
    predicted_phase = predicted_crystal[..., 3].view(img_size, img_size)
    predicted_dp = compute_fft_dp(predicted_crystal)
    predicted_dp_norm = predicted_dp/torch.max(predicted_dp)

    # Generate true crystal image from seed data
    true_crystal = seed_to_crystal_true(true_seed, grid_size=grid_size, pad_size=img_size)
    #print(f"True crystal data sample: {true_seed}")
    true_amplitude = true_crystal[..., 2].view(img_size, img_size).cpu()
    true_phase = true_crystal[..., 3].view(img_size, img_size).cpu()

    # Plot true crystal amplitude and phase
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 3, 1)
    plt.imshow(true_amplitude, cmap='viridis')
    plt.title("True Crystal Amplitude")
    plt.colorbar()
    plt.subplot(2, 3, 2)
    plt.imshow(true_phase, cmap='viridis')
    plt.title("True Crystal Phase")
    plt.colorbar()
    plt.subplot(2, 3, 3)
    plt.imshow(test_diffraction_pattern[..., 2].view(img_size, img_size).cpu(), cmap='jet')
    plt.title("True Diffraction Pattern (No Phase)")
    plt.colorbar()

    # Plot predicted crystal amplitude and phase
    plt.subplot(2, 3, 4)
    plt.imshow(predicted_amplitude, cmap='viridis')
    plt.title("Predicted Crystal Amplitude")
    plt.colorbar()
    plt.subplot(2, 3, 5)
    plt.imshow(predicted_phase, cmap='viridis')
    plt.title("Predicted Crystal Phase")
    plt.colorbar()
    plt.subplot(2, 3, 6)
    plt.imshow(predicted_dp_norm.view(img_size, img_size).cpu(), cmap='jet')
    plt.title("Predicted Diffraction Pattern (No Phase)")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# Initialise the model and train
complex_model = ComplexTransformerModel(input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout, d_model).to(
    device)
optimizer = optim.Adam(complex_model.parameters(), lr=learning_rate)

# Generate data and train the model
complex_model = train_complex_model(complex_model, optimizer, loss_function, train_loader, eval_loader, num_epochs)

# Set the model to evaluation mode
complex_model.eval()

# Get a sample batch from eval_loader (ensure eval_loader has shuffle=False)
eval_data = next(iter(eval_loader))
test_diffraction_pattern, true_seed = eval_data

# Choose the first sample in the batch for evaluation
test_diffraction_pattern = test_diffraction_pattern[0].unsqueeze(0).to(device)  # Add batch dimension
true_seed = true_seed[0].unsqueeze(0).to(device)  # Add batch dimension

# Plot the comparison between predicted and true crystal
plot_comparison(complex_model, test_diffraction_pattern, true_seed, img_size)

