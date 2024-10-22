# Jialun Liu UCL/LCN
# This supervised transformer model has the input: diffraction pattern (x, y, amplitude), the model Output: predicted crystal in the form of (x, y, amplitude, phase)
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
input_dim = 3  # Diffraction pattern (x, y, amplitude)
output_dim = 4  # Predict (x, y, amplitude, phase)
nhead = 4
d_model = 128
num_layers = 6
dim_feedforward = 256
dropout = 0.1
img_size = 20
num_epochs = 200
batch_size = 32
learning_rate = 1e-4


# Function to create density from amplitude and phase
def create_density_function(amplitude, phase):
    real = amplitude * torch.cos(phase)  # Phase is already in radians (-pi to pi)
    imag = amplitude * torch.sin(phase)
    return torch.complex(real, imag)


# Generate a real-space crystal with amplitude and phase
def generate_crystal(img_size=20, grid_size=10):
    crystal = np.zeros((img_size, img_size), dtype=complex)

    center = img_size // 2
    start = center - grid_size // 2
    end = start + grid_size

    amplitude = np.random.rand(grid_size, grid_size)  # Amplitude between 0 and 1
    phase = (np.random.rand(grid_size, grid_size) * 2 * np.pi) - np.pi  # Phase between -pi and pi
    crystal[start:end, start:end] = amplitude * np.exp(1j * phase)

    return crystal


# Generate a diffraction pattern (only amplitude) from a real-space crystal
def generate_diffraction_pattern(crystal, img_size):
    # Perform Fourier transform on the crystal to get the diffraction pattern
    real_space = torch.fft.fftshift(torch.tensor(crystal, dtype=torch.complex64), dim=(-2, -1))
    diffraction_pattern = torch.fft.fft2(real_space)
    diffraction_pattern = torch.fft.fftshift(diffraction_pattern, dim=(-2, -1))

    # Amplitude is the square root of intensity
    amplitude = torch.sqrt(torch.abs(diffraction_pattern))

    # Normalize amplitude between 0 and 1
    amplitude_norm = amplitude / (torch.max(amplitude) + 1e-8)

    # Generate (x, y, amplitude) input for the model
    coords = [(i / img_size, j / img_size, amplitude_norm[i, j].item()) for i in range(img_size) for j in
              range(img_size)]
    coords = torch.tensor(coords, dtype=torch.float32)

    return coords


# Create a dataset of crystal pairs and corresponding diffraction patterns (without phase)
def create_crystal_data(img_size, num_crystals=10):
    data = []

    for _ in range(num_crystals):
        crystal = generate_crystal(img_size)
        diffraction_pattern = generate_diffraction_pattern(crystal, img_size)

        # Prepare the target (real-space crystal: amplitude and phase)
        real_part = np.real(crystal)
        imag_part = np.imag(crystal)

        coords_real = np.array(
            [(i, j, real_part[i, j], np.angle(crystal[i, j])) for i in range(img_size) for j in range(img_size)])
        coords_real = torch.tensor(coords_real, dtype=torch.float32)

        data.append((diffraction_pattern, coords_real))  # input: diffraction pattern, target: real-space crystal

    crystal_dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return crystal_dataloader


# Transformer model that predicts both amplitude and phase of the crystal
class ComplexTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout, d_model):
        super(ComplexTransformerModel, self).__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        # Project input to d_model dimension
        src = self.input_proj(src)

        # Pass through transformer
        output = self.transformer(src, src)

        # Project back to output_dim (x, y, amplitude, phase)
        output = self.fc_out(output)

        # Normalize x and y positions to [0, 1]
        output[..., 0] = torch.sigmoid(output[..., 0])  # Normalise x to [0, 1]
        output[..., 1] = torch.sigmoid(output[..., 1])  # Normalise y to [0, 1]

        # Normalize amplitude to [0, 1]
        output[..., 2] = torch.sigmoid(output[..., 2])  # Amplitude between 0 and 1

        # Phase is within [-pi, pi]
        output[..., 3] = torch.tanh(output[..., 3]) * torch.pi  # Phase between -pi and pi

        return output


def calculate_chi_square(predicted, target):
    batch_size = predicted.shape[0]

    predicted_dp = torch.zeros(batch_size, img_size, img_size, dtype=torch.float32)  # Use float for magnitude
    target_dp = torch.zeros(batch_size, img_size, img_size, dtype=torch.float32)     # Use float for magnitude

    for batch_idx in range(batch_size):
        # Extract amplitude and phase for the predicted crystal
        amplitude_pred = predicted[batch_idx][..., 2].view(img_size, img_size)  # Amplitude is at index 2
        phase_pred = predicted[batch_idx][..., 3].view(img_size, img_size)      # Phase is at index 3
        predicted_real_space = create_density_function(amplitude_pred, phase_pred)

        # Apply FFT and shift
        predicted_real_space = predicted_real_space.view(img_size, img_size)    # Ensure 2D input for FFT
        predicted_real_space = torch.fft.fftshift(predicted_real_space)
        predicted_dp_complex = torch.fft.fft2(predicted_real_space)
        predicted_dp[batch_idx] = torch.abs(torch.fft.fftshift(predicted_dp_complex))  # Get magnitude

        # Same for target
        amplitude_target = target[batch_idx][..., 2].view(img_size, img_size)   # Amplitude is at index 2
        phase_target = target[batch_idx][..., 3].view(img_size, img_size)       # Phase is at index 3
        target_real_space = create_density_function(amplitude_target, phase_target)

        target_real_space = target_real_space.view(img_size, img_size)          # Ensure 2D input for FFT
        target_real_space = torch.fft.fftshift(target_real_space)
        target_dp_complex = torch.fft.fft2(target_real_space)
        target_dp[batch_idx] = torch.abs(torch.fft.fftshift(target_dp_complex))  # Get magnitude

    predicted_norm = predicted_dp / (torch.max(predicted_dp) + 1e-8)
    target_norm = target_dp / (torch.max(target_dp) + 1e-8)

    chi_square = torch.sum((predicted_norm - target_norm) ** 2 / (
                torch.sqrt(torch.sum(predicted_norm ** 2) * torch.sum(target_norm ** 2)) + 1e-8))
    return chi_square


# Training function with plotting
def train_complex_model(model, optimizer, criterion, dataloader, num_epochs):
    loss_values = []
    eval_loss_values = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        loss_values.append(epoch_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()

    return model


# Plot function to compare diffraction patterns and reconstructed crystals
def plot_comparison(model, test_diffraction_pattern, true_crystal, img_size):
    model.eval()

    # Predict the crystal from the diffraction pattern
    predicted_crystal = model(test_diffraction_pattern.to(device)).detach().cpu()

    # Extract amplitude and phase for visualization
    predicted_amplitude = predicted_crystal[..., 2].view(img_size, img_size)
    predicted_phase = predicted_crystal[..., 3].view(img_size, img_size)

    true_amplitude = true_crystal[..., 2].view(img_size, img_size)
    true_phase = true_crystal[..., 3].view(img_size, img_size)

    # Plot diffraction pattern comparison
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(test_diffraction_pattern[..., 2].view(img_size, img_size), cmap='gray')
    plt.title("Input Diffraction Pattern (No Phase)")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_amplitude, cmap='gray')
    plt.title("Predicted Diffraction Pattern (With Phase)")

    # Plot reconstructed crystal amplitude and phase
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(predicted_amplitude, cmap='viridis')
    plt.title("Predicted Crystal Amplitude")
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_phase, cmap='twilight')
    plt.title("Predicted Crystal Phase")

    plt.show()


# Initialize the model and train
complex_model = ComplexTransformerModel(input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout, d_model).to(
    device)
optimizer = optim.Adam(complex_model.parameters(), lr=learning_rate)

# Generate data and train the model
complex_dataloader = create_crystal_data(img_size, 10)
complex_model = train_complex_model(complex_model, optimizer, calculate_chi_square, complex_dataloader, num_epochs)

# Test with a new diffraction pattern (no phase info) and plot results
test_diffraction_pattern, true_crystal = complex_dataloader.dataset[0]
plot_comparison(complex_model, test_diffraction_pattern.unsqueeze(0), true_crystal.unsqueeze(0), img_size)