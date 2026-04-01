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


class AdaptiveTopologicalLoss(nn.Module):
    """
    Adaptive multi-task loss balancing via learned homoscedastic uncertainty.

    Kendall, Gal & Cipolla (2018) "Multi-Task Learning Using Uncertainty
    to Weigh Losses for Scene Geometry and Semantics", CVPR.

    Instead of a manually tuned topo_weight, the model jointly learns
    task-specific log-variances (s_r, s_t) = log(sigma^2) that automatically
    balance reconstruction and topological objectives:

        L = 1/(2*sigma_r^2) * L_recon  +  1/(2*sigma_t^2) * L_topo
            + log(sigma_r) + log(sigma_t)

    Reparameterized as s = log(sigma^2) for numerical stability:

        L = 0.5 * exp(-s_r) * L_recon  +  0.5 * exp(-s_t) * L_topo
            + 0.5 * s_r  +  0.5 * s_t

    The log(sigma) regularizers prevent the trivial solution sigma -> inf
    (which would zero out all losses). At convergence the effective weight
    ratio w_topo / w_recon reflects the intrinsic difficulty ratio of the
    two objectives, removing the need for grid search over topo_weight.

    Supports 'euclidean', 'pearson', and 'cosine' distance metrics.
    """

    DISTANCE_METRICS = ('euclidean', 'pearson', 'cosine')

    def __init__(self, distance_metric='euclidean'):
        super().__init__()
        if distance_metric not in self.DISTANCE_METRICS:
            raise ValueError(f"distance_metric must be one of {self.DISTANCE_METRICS}")
        self.distance_metric = distance_metric
        self.mse = nn.MSELoss()

        # Learnable log-variance parameters: s = log(sigma^2)
        # Initialized to 0  ->  sigma^2 = 1  ->  initial weight = 0.5
        self.log_var_recon = nn.Parameter(torch.zeros(1))
        self.log_var_topo = nn.Parameter(torch.zeros(1))

    def _dist_euclidean(self, x):
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        dist_sq = x_sq + x_sq.T - 2.0 * (x @ x.T)
        return torch.sqrt(torch.clamp(dist_sq, min=1e-8))

    def _dist_pearson(self, x):
        x_c = x - x.mean(dim=1, keepdim=True)
        norms = torch.sqrt(torch.clamp((x_c ** 2).sum(dim=1, keepdim=True), min=1e-8))
        x_n = x_c / norms
        return torch.clamp(1.0 - x_n @ x_n.T, min=0.0)

    def _dist_cosine(self, x):
        norms = torch.sqrt(torch.clamp((x ** 2).sum(dim=1, keepdim=True), min=1e-8))
        x_n = x / norms
        return torch.clamp(1.0 - x_n @ x_n.T, min=0.0)

    def _distance_matrix(self, x):
        if self.distance_metric == 'euclidean':
            return self._dist_euclidean(x)
        elif self.distance_metric == 'pearson':
            return self._dist_pearson(x)
        else:
            return self._dist_cosine(x)

    def forward(self, x_original, x_reconstructed, z_latent):
        recon_loss = self.mse(x_reconstructed, x_original)

        dist_orig = self._distance_matrix(x_original)
        dist_lat = self._distance_matrix(z_latent)
        topo_loss = self.mse(
            dist_lat / dist_lat.max(),
            dist_orig / dist_orig.max(),
        )

        # Uncertainty-weighted combination
        prec_r = torch.exp(-self.log_var_recon)
        prec_t = torch.exp(-self.log_var_topo)

        total_loss = (0.5 * prec_r * recon_loss + 0.5 * self.log_var_recon
                      + 0.5 * prec_t * topo_loss + 0.5 * self.log_var_topo)

        return total_loss, recon_loss, topo_loss

    @torch.no_grad()
    def effective_weights(self):
        """Return (w_recon, w_topo) -- current effective multipliers."""
        return (
            0.5 * torch.exp(-self.log_var_recon).item(),
            0.5 * torch.exp(-self.log_var_topo).item(),
        )
