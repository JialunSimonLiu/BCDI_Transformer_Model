import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from functions import *
#from loss_functions import *
from matplotlib import pyplot as plt
import h5py

# Parameters
vocab_size = 1000
d_model = 512 # Feature size
nhead = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
batch_size = 10
# Number of epochs
epochs = 3

class CrystalDataset(Dataset):
    def __init__(self, file_path):
        self.file_path = file_path
        with h5py.File(file_path, 'r') as file:
            self.keys = list(file.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        with h5py.File(self.file_path, 'r') as file:
            group = file[self.keys[idx]]
            seeds = torch.tensor(group['seeds'][:], dtype=torch.float32)
            diffraction_pattern = torch.tensor(group['diffraction_patterns'][:], dtype=torch.float32)
        return diffraction_pattern, seeds
# source/input (src) is the diffraction pattern data, target/output (tgt) is the real-space data

# Prepare your dataset
dataset = CrystalDataset('data_seeds_dp.h5')
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

    return total_loss / len(loader), outputs


# Training Loop
lossall = []
for epoch in range(epochs):
    loss, output = train(model, loader, optimizer, criterion, device)
    lossall.append(loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
#print(lossall)
plt.plot(lossall)
plt.show()

output = output.detach().cpu().numpy()
print(output)
plt.imshow(output)
plt.show()
# Save your model
torch.save(model.state_dict(), 'crystal_transformer_model.pth')
