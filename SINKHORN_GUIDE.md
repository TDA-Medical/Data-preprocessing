# COAST Loss Notes

COAST = **C**osine **O**ptimized **A**daptive **S**inkhorn **T**ransport

This is a quick guide on the COAST loss added to the TAE pipeline. COAST combines cosine-based pairwise distance matrices with debiased Sinkhorn divergence under adaptive homoscedastic uncertainty weighting (Kendall et al., 2018).

## What's changed

We had some issues with memory and scale, so here's what was done to fix them:
- **Envelope Theorem**: Instead of backpropping through the whole Sinkhorn loop (which was killing the GPU), I just detach the dual variables after it converges. Gradients work out mathematically anyway. Saves like 50x memory.
- **Log-domain**: Sinkhorn in standard exponential space kept underflowing. Moved everything to logsumexp.
- **Scaling (Multiplier)**: Sinkhorn distance is naturally tiny (~0.6) compared to Recon MSE (~10,000). So I added a multiplier to bump it up, otherwise the gradients just get ignored.
- **Random Sweep**: Since we don't know the best multiplier yet, `run_pipeline.sh` just picks 3 random scales between 10^1 and 10^5 and tries them all.

## How to run

Skip the preprocessing if you already ran it:
```bash
bash run_pipeline.sh --skip-preprocess
```

In `TAE/training/train.py`, I added:
- `--sinkhorn`: turns on the COAST loss
- `--topo-multiplier`: scales the topo loss (default 1.0)

For the outputs, `latent_vis.py` will automatically grab all the sweep models and append the multiplier to the file name so we can tell them apart. `visualization.py` will also plot them together.

## The math

COAST loss:
$$\mathcal{L}_{COAST} = \frac{1}{2}e^{-s_r}\mathcal{L}_{recon} + \frac{1}{2}e^{-s_t} \cdot m \cdot \mathcal{L}_{topo}^{Sink} + \frac{1}{2}s_r + \frac{1}{2}s_t$$

Where Topo is the debiased Sinkhorn divergence (bounded >= 0):
Topo = OT(orig, lat) - 0.5 * OT(lat, lat) - 0.5 * OT(orig, orig)
