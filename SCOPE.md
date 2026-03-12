# Quake III World Model — Full Scope Document

## The World Model Pipeline: End to End

### The Problem

We want a neural network that **dreams Quake III Arena**. Given the current frame and the player's button press, predict the next frame. Chain predictions together → a playable "neural game engine" running entirely inside the model's imagination.

Three things needed:
1. **Data**: millions of (frame, action, next_frame) tuples from real gameplay
2. **A model**: that learns (frame + action) → next_frame
3. **Evaluation**: proof that the dream looks and behaves like the real game

---

## Part 1: Data Generation

### Why Data Is Hard (The CS:GO Cautionary Tale)

CS:GO is a commercial game with a proprietary engine (Source). No API exists to send actions and get pixels back programmatically. TeaPearce (IEEE CoG 2022 Best Paper) hacked together a pipeline by screenshotting the game window, reading RAM with hazedumper offsets, and training an inverse dynamics model to recover actions. This produced 700GB across 5,500 games but was fragile, game-version-specific, and required a real display.

Arnie (CS2) parses .dem replay files which record game events, not pixels — so he still needs to replay them through the game client to render frames. This requires an EC2 instance with a GPU for rendering.

**We don't have this problem.** Quake III's engine is open-source (GPL), and DeepMind Lab wraps it into a clean gym environment.

### What Is a Gym Environment?

A standard Python interface that wraps a game:

```python
import gymnasium as gym
env = gym.make("SomeGame-v0")
obs, info = env.reset()          # first frame (numpy array)
for _ in range(10000):
    action = 2                    # e.g., "move forward"
    obs, reward, done, _, info = env.step(action)
    # obs = next frame as numpy array
```

One line → advance the game one tick, get pixels back. Headless (no monitor), thousands of FPS, deterministic, runs on CPU. This is why DOOM and Quake III dominate ML research — their open-source engines enabled gym wrappers. CS:GO/Valorant/Halo don't have this because Valve/Riot/Microsoft never released their engine code.

### DeepMind Lab

DeepMind took the ioquake3 engine (open-source Quake III), heavily modified it, and created DeepMind Lab (2016). Same idea as ViZDoom for DOOM: Python API, headless, fast, step(action) → pixels. Farama Foundation added Shimmy wrappers for standard Gymnasium compatibility.

DeepMind used this for their **FTW agent** (2018) — RL agents that achieved superhuman performance in Quake III Capture the Flag. They ran hundreds of parallel matches on CPU clusters, generating billions of frames.

**For us:** We can use the same infrastructure to generate unlimited training data. Run RL agents or random policies in DeepMind Lab → collect frame+action pairs at thousands of FPS → store as .npz files.

### OpenArena

A community-made free content pack running on ioquake3. 100% GPL — no commercial Quake III assets needed. Same engine, same capabilities, zero legal issues. Good fallback if licensing is a concern.

### Data Collection Plan

**Where:** Local MacBook (CPU-bound, no GPU needed)

**Agent policies (in order of quality):**
1. Random agent — worst quality but instant, good for pipeline smoke test
2. Scripted policy — move forward, shoot when enemy visible
3. PPO agent — train an RL agent first (like GameNGen did), record its full training trajectory from random to competent. Diverse skill levels = better training data.

**Data format:**
- Frames: uint8 RGB numpy arrays at chosen resolution (64x64 to 160x120)
- Actions: discrete integers (move forward, backward, strafe L/R, turn L/R, shoot)
- Storage: compressed .npz chunks or .pt files (DIAMOND-compatible)

**Data scale:**
- Smoke test: 100K frames (~5 min collection, ~500MB)
- Minimum viable: 1M frames (~1 hr collection, ~2-5GB at 84x84)
- Full scale: 10M frames (~several hours, ~20-50GB)

**Data collection does NOT need cloud/GPU.** Emulators are CPU-bound. Upload to RunPod only for training.

### Resolution Decision

| Resolution | Pixels | Reference | Training Time Estimate |
|-----------|--------|-----------|----------------------|
| 64x64 | 12K | DIAMOND Atari | ~2-3 days (4090) |
| 84x84 | 21K | DMLab default | ~3-5 days |
| 128x128 | 49K | — | ~7-10 days |
| 160x120 | 58K | GameNGen agent input | ~10-14 days |

**Start at 84x84** (DMLab default), scale up if results are promising.

---

## Part 2: Model Architecture

### Why Fork DIAMOND

| Property | DIAMOND | GameNGen | Genie |
|----------|---------|----------|-------|
| Params | **4.4M** (Atari) | 860M | 10.7B |
| Train hardware | **1x RTX 4090** | 128 TPU-v5e | 256 TPUv5p |
| Train time | **2.9 days** | Weeks | Weeks |
| Open source | **Yes (MIT)** | No | No |
| Pixel-space | **Yes** | Latent (SD VAE) | Tokenized |
| Tokenizer needed | **No** | Yes (pretrained) | Yes (train) |

DIAMOND is the only architecture that (a) achieves SOTA quality, (b) trains on a single consumer GPU, (c) is fully open-source, and (d) operates directly on pixels with no tokenizer.

### DIAMOND Architecture Overview

A small **U-Net** (2D convolutional encoder-decoder with skip connections) that does **EDM diffusion**:

1. **Input**: concatenation of [noisy_next_frame, past_L_frames] along channel dimension
2. **Action conditioning**: action integer → learned embedding → adaptive group normalization (scale/shift the feature maps)
3. **Training**: add noise to next_frame at random levels, network learns to denoise. Loss = L2 between predicted and clean frame.
4. **Inference**: start from pure Gaussian noise, denoise in 3 Euler steps → predicted next frame

**EDM vs DDPM**: DDPM (used by Stable Diffusion) causes severe compounding errors over long autoregressive rollouts. EDM (Karras et al.) uses preconditioning that maintains unit variance across noise levels, giving much better stability even with 1 denoising step.

### Key Modifications for Quake III

1. **Observation shape**: Change from Atari (64x64x3) to DMLab output resolution
2. **Action space**: Change from Atari (~18 discrete) to DMLab action space (~7-15 discrete)
3. **Frame buffer depth**: May need more context frames for 3D navigation (start with L=4, experiment up to L=16)
4. **Model capacity**: 3D visuals are more complex than Atari sprites — may need to scale U-Net channels (64/128/256 → 128/256/512), ~20-50M params

### Training Plan

- **Hardware**: 1x RTX 4090 on RunPod ($0.20/hr spot or $0.39/hr on-demand)
- **Batch size**: 64 (DIAMOND default, fits in 24GB VRAM)
- **Optimizer**: Adam, LR ~1e-4
- **Training steps**: ~500K-1M
- **Estimated time**: 3-7 days depending on resolution and model size
- **Estimated cost**: $15-55 on RunPod
- **Logging**: W&B (project: `rzhang139/quake3-worldmodel`)

---

## Part 3: Evaluation

### Priority-Ordered Pipeline

**Phase 1 — Minimum Viable (Day 1 of training):**
- PSNR + LPIPS on teacher-forced frames (predicted vs ground truth next frame)
- Side-by-side frame strips logged to W&B every N training steps
- Training loss curve

**Phase 2 — Video-Level Metrics (Week 1):**
- FVD on 16-frame and 32-frame autoregressive rollouts
- Short rollout videos (16-64 frames) logged to W&B
- SSIM as complement to PSNR

**Phase 3 — Game-Specific (Week 2):**
- Action controllability via IDM F1 score: train a small classifier on (frame_t, frame_{t+1}) → action. If it can recover the correct action from generated frames, the model responds to inputs.
- World stability test: execute N forward actions (e.g., 16 turn_lefts), then N inverse (16 turn_rights). Measure LPIPS between initial and final frame. Lower = more stable.
- Diversity score: intra-batch LPIPS to detect mode collapse

**Phase 4 — Demo (Week 3):**
- FastAPI server on GPU, streaming generated frames via WebSocket to browser
- Player presses WASD → actions sent to server → model generates next frame → streamed back
- "Play Quake III inside a neural network's dream"

### Key Metrics Reference

| Metric | What It Measures | Library | Good Value |
|--------|-----------------|---------|------------|
| PSNR (dB) | Pixel-level reconstruction | torchmetrics | >25 (GameNGen: 29.4) |
| LPIPS | Perceptual similarity | `pip install lpips` | <0.3 (GameNGen: 0.249) |
| SSIM | Structural similarity | torchmetrics | >0.8 |
| FVD | Video temporal coherence | common_metrics_on_video_quality | <200 |
| IDM F1 | Action controllability | Custom classifier | >0.7 |

### The Ultimate Test (stretch goal)

Train an RL agent **entirely inside the world model's dream**, then test it in the **real** DeepMind Lab environment. If the dream-trained agent performs well in reality, the world model is faithful. This is DIAMOND's core evaluation approach.

---

## Part 4: Milestones (Breadth-First)

Following the principle: get each part working end-to-end at minimum viable quality, then iterate.

### M0: Infrastructure (this week)
- [ ] Set up project folder, venv, dependencies
- [ ] Verify DeepMind Lab / Shimmy installation
- [ ] Collect 100K frames with random agent
- [ ] Verify data loading pipeline

### M1: First Dream (week 1-2)
- [ ] Fork DIAMOND, adapt to DMLab observation/action space
- [ ] Train on 100K frames at 64x64 for 12-24 hours
- [ ] Generate first rollout videos (even if blurry/bad)
- [ ] Log to W&B

### M2: Decent Quality (week 2-3)
- [ ] Scale to 1M+ frames
- [ ] Tune model size, resolution, training steps
- [ ] PSNR >20, recognizable environments in rollouts
- [ ] Implement basic eval pipeline (PSNR, LPIPS, FVD)

### M3: Playable Dream (week 3-4)
- [ ] Scale to 5-10M frames with PPO-trained agent data
- [ ] PSNR >25, stable 30+ frame rollouts
- [ ] Interactive demo (keyboard → model → browser)
- [ ] Action controllability verified (IDM F1 >0.5)

### M4: Polish & Share (month 2)
- [ ] Full eval suite
- [ ] Project page with side-by-side videos
- [ ] Blog post / Twitter thread
- [ ] Open-source code + weights

---

## Appendix: Landscape of Shooter World Models

| Game | Who | Year | Architecture | Params | Open Source |
|------|-----|------|-------------|--------|------------|
| DOOM | GameNGen (Google) | 2024 | Fine-tuned SD 1.4 | 860M | No |
| DOOM | gameNgen-repro | 2024 | Reproduction | ~860M | Yes |
| DOOM | Ayaan (multiplayer) | 2026 | DiT + AdaLN | — | In progress |
| Quake II | WHAMM (Microsoft) | 2025 | MaskGIT | 750M | No |
| CS:GO | DIAMOND | 2024 | Diffusion U-Net | 381M | Yes |
| CS:GO/CS2 | Arnie | 2026 | VAE + Flow Matching | — | In progress |
| **Quake III** | **Us** | **2026** | **DIAMOND fork** | **~4-50M** | **Yes** |

---

## Appendix: Why These Three Games Have the Best Infrastructure

The games with gym environments are games where someone wrapped an **open-source engine** into the Gymnasium API:

- **DOOM** (1993): id Software released source code in 1997 (GPL). Researchers built ViZDoom (2016). 7,000 FPS, headless, Python API.
- **Quake III** (1999): id Software released source code in 2005 (GPL). DeepMind built DeepMind Lab (2016). Shimmy provides Gymnasium wrappers.
- **CS:GO** (2012): Valve never released Source engine code. No gym env. Data collection requires screen capture hacks or .dem file parsing + rendering.

That single decision — open-sourcing the engine — 20+ years ago is why DOOM and Quake III dominate ML research today.
