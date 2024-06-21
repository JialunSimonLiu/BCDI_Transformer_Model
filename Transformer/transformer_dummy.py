import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from functions import *
from matplotlib import pyplot as plt

# Parameters
vocab_size = 1000
d_model = 512
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
batch_size = 10
# Number of epochs
epochs = 3
"""
class CrystalDataset(Dataset):

    def __init__(self, diffraction_patterns, real_space_data):
        self.diffraction_patterns = diffraction_patterns
        self.real_space_data = real_space_data

    def __len__(self):
        return len(self.diffraction_patterns)

    def __getitem__(self, idx):
        return self.diffraction_patterns[idx], self.real_space_data[idx]
"""
# Dummy dataset
class CrystalDataset(Dataset):
    def __init__(self):
        # Simulating sequences of length 64 (flattened 8x8 diffraction patterns)
        self.src = torch.randint(0, 1000, (100, 64))
        self.tgt = torch.randint(0, 1000, (100, 64))

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.src[idx], self.tgt[idx]

# Prepare your dataset
dataset = CrystalDataset()
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
#train_dataset = CrystalDataset(train_diffraction_patterns, train_real_space_data)
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = CrystalTransformer(vocab_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# Training Function
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for src, tgt in loader:
        src = src.to(device)
        tgt = tgt.to(device)
        # Debug prints
        print(f"Source tensor size: {src.size()}")
        print(f"Target tensor size: {tgt.size()}")

        optimizer.zero_grad()

        # Target input should be offset by one token for the transformer's prediction task
        tgt_input = tgt[:, :-1]
        tgt_real = tgt[:, 1:]

        # Forward pass
        outputs = model(src, tgt_input)
        outputs = outputs.reshape(-1, outputs.shape[-1])
        tgt_real = tgt_real.reshape(-1)
        # Debug print for outputs size
        print(f"Outputs tensor size: {outputs.size()}")

        # Calculate loss
        loss = criterion(outputs, tgt_real)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# Training Loop
lossall = []
for epoch in range(epochs):
    loss = train(model, loader, optimizer, criterion, device)
    lossall.append(loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
#print(lossall)
plt.plot(lossall)
plt.show()
# Save your model
torch.save(model.state_dict(), 'crystal_transformer_model.pth')
