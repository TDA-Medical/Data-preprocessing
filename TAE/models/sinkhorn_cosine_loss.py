import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinkhornCosineLoss(nn.Module):
    """
    Sinkhorn Topological Loss (Cosine distance).
    Using detached duals for f, g to save memory.
    """
    def __init__(self, eps=0.1, num_iters=50, sigma_orig=None, topo_multiplier=1.0):
        super().__init__()
        self.eps = eps
        self.num_iters = num_iters
        self.sigma_orig = sigma_orig
        self.topo_multiplier = topo_multiplier
        
        self.log_sigma_lat = nn.Parameter(torch.zeros(1))
        self.log_var_recon = nn.Parameter(torch.zeros(1))
        self.log_var_topo = nn.Parameter(torch.zeros(1))
        
        self.mse = nn.MSELoss()

    def _dist_cosine(self, x):
        norms = torch.sqrt(torch.clamp((x ** 2).sum(dim=1, keepdim=True), min=1e-8))
        x_n = x / norms
        # Cosine distance: 1 - cosine_similarity
        return torch.clamp(1.0 - x_n @ x_n.T, min=0.0)

    def _sinkhorn_log_domain(self, a, b, C):
        log_a = torch.log(torch.clamp(a, min=1e-8))
        log_b = torch.log(torch.clamp(b, min=1e-8))
        log_C = -C / self.eps
        
        f = torch.zeros_like(a)
        g = torch.zeros_like(b)
        
        log_C_expanded = log_C.unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(self.num_iters):
                g = self.eps * log_b - self.eps * torch.logsumexp(f.unsqueeze(2) + log_C_expanded, dim=1)
                f = self.eps * log_a - self.eps * torch.logsumexp(log_C_expanded + g.unsqueeze(1), dim=2)
            
        cost = torch.sum(a * f, dim=1) + torch.sum(b * g, dim=1)
        return cost.mean()

    def forward(self, x_original, x_reconstructed, z_latent):
        recon_loss = self.mse(x_reconstructed, x_original)
        
        D_orig = self._dist_cosine(x_original)
        D_latent = self._dist_cosine(z_latent)
        
        if self.sigma_orig is None:
            sigma_orig = D_orig.median()
            sigma_orig = torch.where(sigma_orig == 0, torch.tensor(1e-6, device=D_orig.device), sigma_orig)
        else:
            sigma_orig = self.sigma_orig
            
        sigma_lat = F.softplus(self.log_sigma_lat) + 1e-6
        
        S_orig = torch.exp(- (D_orig ** 2) / (2 * sigma_orig ** 2))
        S_lat = torch.exp(- (D_latent ** 2) / (2 * sigma_lat ** 2))
        
        mask = 1.0 - torch.eye(S_orig.size(0), device=S_orig.device)
        S_orig = S_orig * mask
        S_lat = S_lat * mask
        
        P_orig = S_orig / torch.clamp(S_orig.sum(dim=1, keepdim=True), min=1e-8)
        P_lat = S_lat / torch.clamp(S_lat.sum(dim=1, keepdim=True), min=1e-8)
        
        C = D_orig
        
        ot_orig_lat = self._sinkhorn_log_domain(P_orig, P_lat, C)
        ot_lat_lat = self._sinkhorn_log_domain(P_lat, P_lat, C)
        ot_orig_orig = self._sinkhorn_log_domain(P_orig, P_orig, C)
        
        # Debiased Sinkhorn Divergence: S(a,b) = OT(a,b) - 0.5*OT(a,a) - 0.5*OT(b,b)
        # This is guaranteed to be >= 0 and S(a,a) = 0.
        topo_loss = torch.clamp(ot_orig_lat - 0.5 * ot_lat_lat - 0.5 * ot_orig_orig, min=0.0)
        topo_loss = topo_loss * self.topo_multiplier
        
        prec_r = torch.exp(-self.log_var_recon)
        prec_t = torch.exp(-self.log_var_topo)
        
        total_loss = (0.5 * prec_r * recon_loss + 0.5 * self.log_var_recon
                      + 0.5 * prec_t * topo_loss + 0.5 * self.log_var_topo)
                      
        return total_loss, recon_loss, topo_loss
