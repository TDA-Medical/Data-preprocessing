import torch
import torch.nn as nn


class TopologicalLoss(nn.Module):
    """
    Combined loss for the Topological Autoencoder.

    Total loss = reconstruction_loss + topo_weight * topological_loss

    The topological loss measures how well the latent space preserves
    pairwise distance relationships from the original space, by comparing
    normalized distance matrices (MSE between them).
    """

    def __init__(self, topo_weight=1.0):
        super().__init__()
        self.topo_weight = topo_weight
        self.mse = nn.MSELoss()

    def _distance_matrix(self, x):
        """Compute pairwise Euclidean distance matrix for a batch."""
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        dist_sq = x_sq + x_sq.T - 2.0 * (x @ x.T)
        return torch.sqrt(torch.clamp(dist_sq, min=1e-8))

    def forward(self, x_original, x_reconstructed, z_latent):
        recon_loss = self.mse(x_reconstructed, x_original)

        dist_original = self._distance_matrix(x_original)
        dist_latent = self._distance_matrix(z_latent)

        # Normalize distances to [0, 1] so the two spaces are comparable
        dist_original_norm = dist_original / dist_original.max()
        dist_latent_norm = dist_latent / dist_latent.max()
        topo_loss = self.mse(dist_latent_norm, dist_original_norm)

        total_loss = recon_loss + self.topo_weight * topo_loss
        return total_loss, recon_loss, topo_loss
