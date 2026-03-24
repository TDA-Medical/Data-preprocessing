"""
GPU-accelerated ComBat batch effect correction using PyTorch.
Drop-in replacement for neuroCombat. Optimized for 8GB VRAM (float32, in-place ops).
"""

import torch
import numpy as np
import pandas as pd


def gpu_combat(dat, covars, batch_col, categorical_cols=None, continuous_cols=None, device=None):
    """
    Remove batch effects from gene expression data using the ComBat algorithm on GPU.

    Args:
        dat:              Expression matrix, shape (n_features, n_samples).
        covars:           DataFrame with batch and covariate columns (one row per sample).
        batch_col:        Column name in covars identifying the batch.
        categorical_cols: Categorical covariates to preserve (default: none).
        continuous_cols:  Continuous covariates to preserve (default: none).
        device:           Torch device (default: CUDA if available).

    Returns:
        Dict with 'data' key: corrected matrix as np.ndarray, shape (n_features, n_samples).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    categorical_cols = categorical_cols or []
    continuous_cols = continuous_cols or []
    dtype = torch.float32

    print(f"[GPU ComBat] Using device: {device} (float32)")

    # --- Input ---
    if isinstance(dat, pd.DataFrame):
        dat_np = dat.values.astype(np.float32)
    else:
        dat_np = np.array(dat, dtype=np.float32)
    n_features, n_samples = dat_np.shape

    # --- 1. Design matrix ---
    print("[GPU ComBat] Creating design matrix")

    batch_labels = covars[batch_col].values
    unique_batches = np.unique(batch_labels)
    n_batch = len(unique_batches)
    batch_map = {b: i for i, b in enumerate(unique_batches)}
    batch_idx = np.array([batch_map[b] for b in batch_labels])
    batch_info = [np.where(batch_idx == i)[0] for i in range(n_batch)]
    sample_per_batch = np.array([len(bi) for bi in batch_info])

    # One-hot encode batches
    batch_onehot = np.zeros((n_samples, n_batch), dtype=np.float32)
    for i in range(n_batch):
        batch_onehot[batch_info[i], i] = 1.0
    design_parts = [batch_onehot]

    # One-hot encode categorical covariates (drop first level to avoid collinearity)
    for col in categorical_cols:
        vals = covars[col].values
        levels = np.unique(vals)
        onehot = np.zeros((n_samples, len(levels)), dtype=np.float32)
        level_map = {v: i for i, v in enumerate(levels)}
        for j, v in enumerate(vals):
            onehot[j, level_map[v]] = 1.0
        design_parts.append(onehot[:, 1:])

    # Continuous covariates as-is
    for col in continuous_cols:
        design_parts.append(covars[col].values.astype(np.float32).reshape(-1, 1))

    design = torch.tensor(np.hstack(design_parts), dtype=dtype, device=device)

    # --- 2. Standardize ---
    print("[GPU ComBat] Standardizing data across features")

    X = torch.tensor(dat_np, dtype=dtype, device=device)
    del dat_np

    # Fit OLS: gene_expression = design @ beta
    B_hat = torch.linalg.lstsq(design, X.T).solution

    # Grand mean: weighted average of batch coefficients
    weights = torch.tensor(sample_per_batch / n_samples, dtype=dtype, device=device)
    grand_mean = weights @ B_hat[:n_batch, :]

    # Covariate-only effects (batch columns zeroed)
    design_covars_only = design.clone()
    design_covars_only[:, :n_batch] = 0.0
    covar_effects = (design_covars_only @ B_hat).T
    del design_covars_only

    # Pooled variance from residuals (in-place to save VRAM)
    residuals = (design @ B_hat).T
    residuals.sub_(X).mul_(-1)
    var_pooled = (residuals ** 2).mean(dim=1, keepdim=True)
    del residuals
    torch.cuda.empty_cache()

    # Replace zero variances with median
    nonzero = var_pooled.squeeze() > 0
    if (~nonzero).any():
        var_pooled[~nonzero] = var_pooled[nonzero].median()

    # Standardize in-place: s = (X - grand_mean - covar_effects) / sqrt(var_pooled)
    X.sub_(grand_mean.unsqueeze(1))
    X.sub_(covar_effects)
    X.div_(var_pooled.sqrt())
    s_data = X

    # --- 3. Estimate batch effects ---
    print("[GPU ComBat] Fitting L/S model and finding priors")

    batch_design = design[:, :n_batch]
    gamma_hat = torch.linalg.lstsq(batch_design, s_data.T).solution  # location shifts

    # Scale factors: within-batch variance per gene
    delta_hat = torch.ones(n_batch, n_features, dtype=dtype, device=device)
    for i in range(n_batch):
        delta_hat[i] = s_data[:, batch_info[i]].var(dim=1)
    delta_hat.clamp_(min=1e-10)

    # Empirical Bayes hyperparameters
    gamma_bar = gamma_hat.mean(dim=0)
    t2 = gamma_hat.var(dim=0)

    a_prior = torch.zeros(n_batch, dtype=dtype, device=device)
    b_prior = torch.zeros(n_batch, dtype=dtype, device=device)
    for i in range(n_batch):
        m = delta_hat[i].mean()
        s2 = delta_hat[i].var().clamp(min=1e-10)
        a_prior[i] = (2 * s2 + m ** 2) / s2
        b_prior[i] = (m * s2 + m ** 3) / s2

    # --- 4. Empirical Bayes shrinkage ---
    print("[GPU ComBat] Finding parametric adjustments")

    gamma_star = torch.zeros_like(gamma_hat)
    delta_star = torch.zeros_like(delta_hat)

    for i in range(n_batch):
        sdat = s_data[:, batch_info[i]]
        n_b = len(batch_info[i])
        g_old = gamma_hat[i].clone()
        d_old = delta_hat[i].clone()

        for _ in range(1000):
            # Posterior mean for location
            g_new = (t2 * n_b * gamma_hat[i] + d_old * gamma_bar) / (t2 * n_b + d_old)
            # Posterior mean for scale (inverse-gamma)
            sum2 = ((sdat - g_new.unsqueeze(1)) ** 2).sum(dim=1)
            d_new = (0.5 * sum2 + b_prior[i]) / (n_b / 2.0 + a_prior[i] - 1.0)

            g_change = ((g_new - g_old).abs() / (g_old.abs() + 1e-10)).max()
            d_change = ((d_new - d_old).abs() / (d_old.abs() + 1e-10)).max()
            g_old, d_old = g_new, d_new

            if max(g_change.item(), d_change.item()) < 0.0001:
                break

        gamma_star[i] = g_new
        delta_star[i] = d_new

        if (i + 1) % 100 == 0:
            print(f"  EB convergence: {i + 1}/{n_batch} batches")

    print(f"  EB convergence: {n_batch}/{n_batch} batches")

    # --- 5. Apply correction ---
    print("[GPU ComBat] Adjusting data")

    for i in range(n_batch):
        idx = batch_info[i]
        s_data[:, idx] = (s_data[:, idx] - gamma_star[i].unsqueeze(1)) / delta_star[i].sqrt().unsqueeze(1)

    # Restore original scale: corrected = s_data * sqrt(var) + grand_mean + covar_effects
    s_data.mul_(var_pooled.sqrt())
    s_data.add_(grand_mean.unsqueeze(1))
    s_data.add_(covar_effects)

    result = s_data.cpu().numpy()

    del s_data, design, B_hat, gamma_hat, delta_hat, gamma_star, delta_star
    del covar_effects, var_pooled, batch_design
    torch.cuda.empty_cache()

    print("[GPU ComBat] Done!")
    return {'data': result}
