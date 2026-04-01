# Sinkhorn Topo Loss Notes

This is a quick guide on the new Sinkhorn Optimal Transport (OT) topological loss added to the TAE pipeline.

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
- `--sinkhorn`: turns on the OT loss
- `--topo-multiplier`: scales the topo loss (default 1.0)

For the outputs, `latent_vis.py` will automatically grab all the sweep models and append the multiplier to the file name so we can tell them apart. `visualization.py` will also plot them together.

## The math

Basically:
Total Loss = Recon + (Multiplier * Topo) + Adaptive Uncertainty terms

Where Topo is the debiased version so it doesn't drop below 0:
Topo = OT(orig, lat) - 0.5 * OT(lat, lat) - 0.5 * OT(orig, orig)
