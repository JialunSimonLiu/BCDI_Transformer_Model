import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from functions import *

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
        # src is the diffraction pattern info
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
        seed_predictions[..., 3] = torch.tanh(seed_predictions[..., 3]) * torch.pi  # Phase normalised to [-π, π]
        #((seed_predictions[..., 3] + 1) * 0.5) * 2 * torch.pi - torch.pi
        #print(seed_predictions)
        # Unnormalise positions for Voronoi computation
        unnormalised_positions = unnormalise_positions(seed_predictions, grid_size=grid_size)
        # Seed to crystal image (pixels), shape (batch, num_seeds, 4)
        crystal_image = seed_to_crystal(unnormalised_positions, grid_size=grid_size, pad_size=img_size)

        return crystal_image


"""class ComplexTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, nhead, num_layers, dim_feedforward, dropout, d_model, num_seeds=6):
        super(ComplexTransformerModel, self).__init__()
        self.num_seeds = num_seeds

        # Input projection layer for DP
        self.input_proj = nn.Linear(input_dim, d_model)

        # Learnable seed query embeddings for predicting seeds
        self.seed_queries = nn.Parameter(torch.randn(1, num_seeds, d_model))

        # Positional encodings to help model seed positions
        self.seed_positional_encoding = nn.Parameter(torch.randn(1, num_seeds, d_model))

        # Main transformer layer with dropout for regularization
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, batch_first=True)

        # Cross-seed attention for regularization with dropout
        self.cross_seed_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # Linear layer to project transformer output to (x, y, amplitude, phase) for each seed
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src):
        batch_size = src.size(0)

        # Project DP input to d_model dimensions
        dp_embed = self.input_proj(src)

        # Expand seed queries and add positional encoding
        seed_queries = self.seed_queries.expand(batch_size, -1, -1) + self.seed_positional_encoding
        combined_input = torch.cat([dp_embed, seed_queries], dim=1)

        # Transformer processing with combined input
        combined_output = self.transformer(combined_input, combined_input)

        # Extract seed predictions by selecting only the output for seed query positions
        seed_output = combined_output[:, -self.num_seeds:, :]

        # Apply cross-seed attention for more robust separation between seed predictions
        seed_output, _ = self.cross_seed_attention(seed_output, seed_output, seed_output)

        # Project to (x, y, amplitude, phase) for each seed
        seed_predictions = self.fc_out(seed_output)

        # Normalizing x, y, amplitude to [0, 1] and phase to [-π, π] for stability
        seed_predictions[..., 0:3] = torch.sigmoid(seed_predictions[..., 0:3])
        seed_predictions[..., 3] = torch.tanh(seed_predictions[..., 3]) * torch.pi

        # Unnormalise positions for Voronoi computation
        unnormalised_positions = unnormalise_positions(seed_predictions, grid_size=grid_size)
        # Seed to crystal image (pixels), shape (batch, num_seeds, 4)
        crystal_image = seed_to_crystal(unnormalised_positions, grid_size=grid_size, pad_size=img_size)

        return crystal_image
"""