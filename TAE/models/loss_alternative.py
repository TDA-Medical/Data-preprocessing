import torch
import torch.nn as nn


class PearsonTopologicalLoss(nn.Module):
    """
    Topological loss using Pearson correlation distance.

    Total loss = reconstruction_loss + topo_weight * topological_loss

    The topological loss compares normalized pairwise distance matrices,
    where distance is defined as  D_ij = 1 - r_ij  (Pearson correlation
    distance).  Pearson correlation measures linear co-expression patterns
    between samples, making it robust to differences in absolute magnitude
    and more suitable for high-dimensional gene expression data where
    Euclidean distance suffers from the curse of dimensionality.
    """

    def __init__(self, topo_weight=1.0):
        super().__init__()
        self.topo_weight = topo_weight
        self.mse = nn.MSELoss()

    def _distance_matrix(self, x):
        """Compute pairwise Pearson correlation distance matrix for a batch.

        D_ij = 1 - r_ij

        where r_ij is the Pearson correlation coefficient between samples i and j.
        Equivalent to cosine similarity on mean-centered vectors.
        """
        # Mean-center each sample (row-wise)
        x_centered = x - x.mean(dim=1, keepdim=True)

        # Compute norms of centered vectors
        norms = torch.sqrt(torch.clamp((x_centered ** 2).sum(dim=1, keepdim=True), min=1e-8))

        # Pearson correlation = cosine similarity of centered vectors
        x_norm = x_centered / norms
        corr = x_norm @ x_norm.T

        # Convert to distance: D = 1 - r, clamp for numerical stability
        return torch.clamp(1.0 - corr, min=0.0)

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


class CosineTopologicalLoss(nn.Module):
    """
    Topological loss using cosine distance.

    Total loss = reconstruction_loss + topo_weight * topological_loss

    The topological loss compares normalized pairwise distance matrices,
    where distance is defined as  D_ij = 1 - cos(x_i, x_j).  Cosine
    distance measures the angular divergence between sample vectors,
    ignoring magnitude differences.  This captures directional similarity
    in gene expression profiles, which is useful when relative expression
    patterns matter more than absolute levels.
    """

    def __init__(self, topo_weight=1.0):
        super().__init__()
        self.topo_weight = topo_weight
        self.mse = nn.MSELoss()

    def _distance_matrix(self, x):
        """Compute pairwise cosine distance matrix for a batch.

        D_ij = 1 - cos(x_i, x_j)
             = 1 - (x_i . x_j) / (||x_i|| * ||x_j||)
        """
        # Compute L2 norms per sample
        norms = torch.sqrt(torch.clamp((x ** 2).sum(dim=1, keepdim=True), min=1e-8))

        # Normalize to unit vectors
        x_norm = x / norms

        # Cosine similarity matrix
        cos_sim = x_norm @ x_norm.T

        # Convert to distance: D = 1 - cos_sim, clamp for numerical stability
        return torch.clamp(1.0 - cos_sim, min=0.0)

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
