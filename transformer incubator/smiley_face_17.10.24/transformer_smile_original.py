# Jialun Liu LCN 17.10.2024
# The original transformer model to reconstruct a smiley face

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
input_dim = 3  # (x, y, intensity)
output_dim = 3  # Predict (x, y, intensity)
nhead = 4  # Increased number of heads for better attention
d_model = 128  # Increased model dimension
num_layers = 6  # Increased the number of layers
dim_feedforward = 256  # Increased feedforward dimension for more complex patterns
dropout = 0.1
img_size = 20  # Assuming square image
num_epochs = 200
batch_size = 32
learning_rate = 1e-4
use_mse_loss = True  # Switch between chi-square and MSE loss


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


# Chi-square loss function
def calculate_chi_square(predicted, target):
    predicted_intensity = predicted[..., 2]
    target_intensity = target[..., 2]

    # Normalize intensities and avoid division by zero
    predicted_norm = predicted_intensity / (torch.max(predicted_intensity) + 1e-8)
    target_norm = target_intensity / (torch.max(target_intensity) + 1e-8)

    # Chi-squared computation
    chi_square = torch.sum((predicted_norm - target_norm) ** 2 / (target_norm + 1e-8))
    return chi_square


# Example dataset creation (dummy data)
def create_dummy_data(img_size, num_samples):
    X = torch.rand(num_samples, img_size * img_size, 3)  # Random (x, y, intensity)
    Y = X.clone()  # Target is same as input for now (identity)
    return [(X, Y) for _ in range(num_samples)]

def create_smiley_face_data(img_size, num_samples):
    X = torch.zeros(num_samples, img_size * img_size, 3)  # Initialize with zeros (black background)
    intensity = 1.0

    # Define positions of the eyes and mouth as a list of coordinates
    eyes = [(5, 6), (5, 13)]  # Two eyes
    mouth = [(12, 8), (13, 9), (13, 10), (13, 11), (12, 12)]  # Simple arc for the mouth

    # Set intensity for eyes
    for eye in eyes:
        X[0, eye[0] * img_size + eye[1], 2] = intensity

    # Set intensity for mouth
    for m in mouth:
        X[0, m[0] * img_size + m[1], 2] = intensity

    # Set (x, y) normalized values
    X[0, :, 0] = torch.linspace(0, 1, img_size).repeat(img_size)  # x-coordinate normalized
    X[0, :, 1] = torch.repeat_interleave(torch.linspace(0, 1, img_size), img_size)  # y-coordinate normalized

    return [(X, X.clone())]  # Use the same data as target




# Plot comparison between predicted and ground truth
def plot_comparison(predicted, ground_truth, img_size):
    predicted = predicted.cpu().detach().numpy().reshape(img_size, img_size, 3)
    ground_truth = ground_truth.cpu().detach().numpy().reshape(img_size, img_size, 3)

    # Plotting predicted and ground truth side by side
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(predicted[..., 2], cmap='gray')  # Predicted intensity
    ax[0].set_title('Predicted Intensity')

    ax[1].imshow(ground_truth[..., 2], cmap='gray')  # Ground truth intensity
    ax[1].set_title('Ground Truth Intensity')

    plt.show()


# Training loop with loss tracking
def train_model(model, optimizer, criterion, dataloader, num_epochs):
    loss_values = []  # List to store loss values

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()  # Accumulate loss

        epoch_loss = running_loss / len(dataloader)  # Average loss for the epoch
        loss_values.append(epoch_loss)  # Store epoch loss
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Plot loss after training
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting after final epoch
    inputs, targets = next(iter(dataloader))  # Get one batch
    outputs = model(inputs.to(device))
    plot_comparison(outputs, targets, img_size)


# Initialize model, optimizer, and loss function
model = TransformerModel(input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout, d_model).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = calculate_chi_square if not use_mse_loss else nn.MSELoss()

# Create smiley face data
dataloader = create_smiley_face_data(img_size, 1)

# Train the model
train_model(model, optimizer, criterion, dataloader, num_epochs)
