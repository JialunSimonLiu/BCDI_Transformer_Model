import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

grid_size = 16  # Size of the crystal (for Voronoi diagram)
img_size = 32  # Padded size (output image size)
N_seed = 6

def unnormalise_positions(seeds, grid_size=grid_size):
    # Unnormalizes (x, y) from [0, 1] to [0, grid_size-1] and keeps all features
    unnormalised_seeds = seeds.clone()
    unnormalised_seeds[..., :2] = (seeds[..., :2] * (grid_size - 1)).round().int()
    return unnormalised_seeds


def discrete_voronoi(seeds, grid_size=grid_size):
    # Batch handling for Voronoi computation
    batch_size, num_seeds, _ = seeds.size()
    device = seeds.device

    # Creating a meshgrid for the coordinates and adjusting for batch compatibility
    x = torch.arange(0, grid_size, device=device)
    y = torch.arange(0, grid_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')

    # Expanding dimensions to match batch and seed dimensions
    xx = xx.unsqueeze(0).unsqueeze(-1).expand(batch_size, grid_size, grid_size, num_seeds)
    yy = yy.unsqueeze(0).unsqueeze(-1).expand(batch_size, grid_size, grid_size, num_seeds)

    # Calculate distances from each grid point to each seed in the batch
    distances = torch.sqrt((xx - seeds[:, :, 0].view(batch_size, 1, 1, num_seeds)) ** 2 +
                           (yy - seeds[:, :, 1].view(batch_size, 1, 1, num_seeds)) ** 2)

    # Find the nearest seed index for each point in the grid
    nearest_seed_index = torch.argmin(distances, dim=3)
    return nearest_seed_index


def assign_phase_values(nearest_seed_index, seeds, grid_size=grid_size, pad_size=img_size):
    device = seeds.device
    batch_size, num_seeds, _ = seeds.size()

    # Extract amplitude and phase
    amplitude_values = seeds[:, :, 2]
    phase_values = seeds[:, :, 3]
    pred_phases = ((phase_values + torch.pi) % (2 * torch.pi)) - torch.pi

    # Assign amplitude and phase values
    amplitude_assigned = amplitude_values.gather(1, nearest_seed_index.view(batch_size, -1))
    phase_assigned = pred_phases.gather(1, nearest_seed_index.view(batch_size, -1))

    # Reshape and pad to (pad_size, pad_size)
    amplitude_assigned = amplitude_assigned.view(batch_size, grid_size, grid_size)
    phase_assigned = phase_assigned.view(batch_size, grid_size, grid_size)
    pad_width = (pad_size - grid_size) // 2
    amplitude_assigned = F.pad(amplitude_assigned, (pad_width, pad_width, pad_width, pad_width), mode='constant',
                               value=0)
    phase_assigned = F.pad(phase_assigned, (pad_width, pad_width, pad_width, pad_width), mode='constant', value=0)

    # Create circular mask and apply
    radius, circle_center = grid_size // 2, pad_size // 2
    x = torch.arange(0, pad_size, device=device)
    y = torch.arange(0, pad_size, device=device)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    mask = ((xx - circle_center) ** 2 + (yy - circle_center) ** 2 > radius ** 2).view(1, -1).repeat(batch_size, 1)
    amplitude_assigned = amplitude_assigned.view(batch_size, -1).masked_fill(mask, 0)
    phase_assigned = phase_assigned.view(batch_size, -1).masked_fill(mask, 0)

    # Flatten and stack for final output
    x_positions = xx.flatten().unsqueeze(0).repeat(batch_size, 1)
    y_positions = yy.flatten().unsqueeze(0).repeat(batch_size, 1)
    crystal_image = torch.stack((x_positions, y_positions, amplitude_assigned, phase_assigned), dim=-1)

    # Extract phase for plotting (the first image in the batch)
    phase_data = crystal_image[0, :, 3].view(pad_size, pad_size).cpu().detach().numpy()
    return crystal_image


def seed_to_crystal(seed_predictions, grid_size=grid_size, pad_size=img_size):
    # Unnormalise positions for Voronoi computation
    unnormalised_positions = unnormalise_positions(seed_predictions, grid_size=grid_size)

    # Generate the Voronoi diagram based on nearest seed positions
    nearest_seed_index = discrete_voronoi(unnormalised_positions, grid_size=grid_size)

    # Assign amplitude and phase values based on Voronoi regions and create the final padded crystal image
    crystal_image = assign_phase_values(nearest_seed_index, unnormalised_positions, grid_size=grid_size,
                                        pad_size=pad_size)

    return crystal_image


class ComplexTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout, d_model, num_seeds=6):
        super(ComplexTransformerModel, self).__init__()
        self.num_seeds = num_seeds

        # Input projection layer for DP
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable seed query embeddings for predicting seeds
        self.seed_queries = nn.Parameter(torch.randn(1, num_seeds, d_model))
        self.expanded_seed_queries = self.seed_queries  # Remove repetitive expand in forward

        # Transformer layer
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)

        # Linear layer to project transformer output to (x, y, amplitude, phase) for each seed
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        batch_size = src.size(0)

        # Project DP input to d_model dimensions
        dp_embed = self.input_proj(src)  # Shape: (batch_size, dp_src_size, d_model)

        # Expand and concatenate learnable seed queries with DP embeddings
        seed_queries = self.seed_queries.expand(batch_size, -1, -1)  # Shape: (batch_size, num_seeds, d_model)
        combined_input = torch.cat([dp_embed, seed_queries], dim=1)  # Shape: (batch_size, dp_src_size + num_seeds, d_model)

        # Transformer processing
        combined_output = self.transformer(combined_input,combined_input)  # Shape: (batch_size, dp_src_size + num_seeds, d_model)

        # Extract seed predictions by selecting only the output for seed query positions
        seed_output = combined_output[:, -self.num_seeds:, :]  # Shape: (batch_size, num_seeds, d_model)

        # Project to (x, y, amplitude, phase) for each seed
        seed_predictions = self.fc_out(seed_output)  # Shape: (batch_size, num_seeds, 4)

        # Ensure positive outputs for x, y, and amplitude, and phase within [-π, π]
        seed_predictions[..., 0:3] = torch.sigmoid(seed_predictions[..., 0:3])  # Normalise x, y, amplitude to [0, 1]
        #seed_predictions[..., 3] = torch.tanh(seed_predictions[..., 3]) * torch.pi  # Phase normalised to [-π, π]
        #print(seed_predictions)

        # Seed to crystal image (pixels)
        # Assuming `seed_predictions` is output from the model with shape (batch, num_seeds, 4)
        crystal_image = seed_to_crystal(seed_predictions, grid_size=grid_size, pad_size=img_size)
        return crystal_image  # Should output crystal full image (batch_size, image_size*image_size, 4)