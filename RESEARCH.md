# Research Log

## 2026-03-12: DoomMWM — Multiplayer World Model

### Problem
Current multiplayer setup (predicting two egocentric agent views concatenated in pixel space) shows excessive hallucination and world-state inconsistency.

### Hypothesis
Concatenated multi-view pixel prediction is under-constrained as a world-modeling objective. It encourages locally plausible per-view reconstructions rather than a single coherent latent state explaining both agents consistently across time.

The model learns **two plausible synchronized videos** rather than a **shared dynamical world model**. This is the fundamental downfall of latents trained on pixel reconstruction loss (RIP 3D-VAE), especially at lower resolution.

Bottleneck is inductive bias toward shared state, not raw optimization.

### Plan
1. **Pretrain strong single-view Doom world model at high resolution** — internalize core Doom dynamics, persistence, interaction structure first. In progress (~5K steps).
2. **Mid-train multiplayer**: concatenation baseline → lightweight agent-specific DiT heads. Single-view curriculum teaches game dynamics before tackling multi-agent observation consistency.

---

## 2026-03-12: Single-Player Baseline (this repo)

First end-to-end training run. EDM diffusion U-Net (small, 2.9M params) on 1000 ViZDoom deathmatch episodes.

- **Config**: batch=128, lr=4e-4, noise_aug=0.1, 100 epochs
- **Result**: loss 0.68 → 0.032 over 141.9K steps (~14h on RTX A5000)
- **Cost**: ~$3.10
- **W&B**: https://wandb.ai/rzhang139/quake3-worldmodel/runs/hu2mby3e
- **Next**: eval rollouts, scale to medium model, higher resolution
