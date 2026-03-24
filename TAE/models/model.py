import torch
import torch.nn as nn


class TopologicalAutoencoder(nn.Module):
    """
    Autoencoder for RNA-seq data that preserves topological structure
    in the latent space. Supports latent dimensions of 8, 16, or 32.

    Architecture:
        Encoder: input_dim -> 1024 -> 256 -> latent_dim
        Decoder: latent_dim -> 256 -> 1024 -> input_dim
    """

    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, input_dim),
            nn.ReLU()  # TPM values are non-negative
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z
