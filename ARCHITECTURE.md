# World Model Architecture

## Overview

Given the last 4 frames + player action, predict the next frame using EDM diffusion.

```
INPUTS                          CONDITIONING                    OUTPUT
──────                          ────────────                    ──────

Context frames (4x)             Action (int)
  ┌─────┐┌─────┐                   │
  │ t-3 ││ t-2 │...                ▼
  └─────┘└─────┘              ┌──────────┐
       │                      │ Embedding │  (lookup table: action → 256-dim vector)
       │                      │ 10 × 256  │
       │                      └────┬─────┘
       │                           │
       │         Noise level σ     │
       │         (scalar)          │
       │            │              │
       │            ▼              │
       │     ┌─────────────┐       │
       │     │   Fourier    │      │
       │     │  Features    │      │
       │     │ σ → 64-dim   │      │
       │     └──────┬──────┘       │
       │            │              │
       │            └──────┬───────┘
       │                   │
       │                   ▼
       │            ┌─────────────┐
       │            │  Concat +   │
       │            │  MLP (2-layer)
       │            │  → 256-dim  │
       │            └──────┬──────┘
       │                   │
       │               cond vector (256-dim)
       │                   │ (fed to every ResBlock via AdaGroupNorm)
       │                   │
       ▼                   ▼
  ┌─────────────────────────────────────────────┐
  │                  U-NET                       │
  │                                              │
  │  ┌───────────────────────────────────────┐   │
  │  │            ENCODER                    │   │
  │  │                                       │   │
  │  │  Input: [noisy_target | ctx frames]   │   │
  │  │         (15 channels = 3 + 4×3)       │   │
  │  │              │                        │   │
  │  │              ▼                        │   │
  │  │  ┌──────────────────────┐             │   │
  │  │  │ conv_in 15ch → 64ch  │             │   │
  │  │  └──────────┬───────────┘             │   │
  │  │             │                         │   │
  │  │    Level 0  │  84×84, 64ch            │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ ResBlock × 2    │◄── cond        │   │
  │  │    └────────┬────────┘  ──────►skip0  │   │
  │  │             │                         │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ Downsample      │                │   │
  │  │    │ (stride-2 conv) │                │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  │    Level 1  │  42×42, 64ch            │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ ResBlock × 2    │◄── cond        │   │
  │  │    └────────┬────────┘  ──────►skip1  │   │
  │  │             │                         │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ Downsample      │                │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  │    Level 2  │  21×21, 64ch            │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ ResBlock × 2    │◄── cond        │   │
  │  │    └────────┬────────┘  ──────►skip2  │   │
  │  │             │                         │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ Downsample      │                │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  │    Level 3  │  11×11, 64ch            │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ ResBlock × 2    │◄── cond        │   │
  │  │    └────────┬────────┘  ──────►skip3  │   │
  │  │             │                         │   │
  │  └─────────────┼─────────────────────────┘   │
  │                │                              │
  │  ┌─────────────┼─────────────────────────┐   │
  │  │         MIDDLE BLOCK                  │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ ResBlock        │◄── cond        │   │
  │  │    │ ResBlock        │◄── cond        │   │
  │  │    └────────┬────────┘                │   │
  │  └─────────────┼─────────────────────────┘   │
  │                │                              │
  │  ┌─────────────┼─────────────────────────┐   │
  │  │            DECODER                    │   │
  │  │             │                         │   │
  │  │    Level 3  │  11×11                  │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ ResBlock × 2    │◄── cond        │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ Upsample        │                │   │
  │  │    │ (nearest + conv)│                │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  │    Level 2  ├◄──── concat skip2       │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ ResBlock × 2    │◄── cond        │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ Upsample        │                │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  │    Level 1  ├◄──── concat skip1       │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ ResBlock × 2    │◄── cond        │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ Upsample        │                │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  │    Level 0  ├◄──── concat skip0       │   │
  │  │    ┌────────┴────────┐                │   │
  │  │    │ ResBlock × 2    │◄── cond        │   │
  │  │    └────────┬────────┘                │   │
  │  │             │                         │   │
  │  └─────────────┼─────────────────────────┘   │
  │                │                              │
  │       ┌────────┴────────┐                     │
  │       │ GroupNorm + SiLU │                     │
  │       │ conv_out 64→3ch  │                     │
  │       └────────┬────────┘                     │
  │                │                              │
  └────────────────┼──────────────────────────────┘
                   │
                   ▼  F(x) = raw network output
            ┌──────────────┐
            │     EDM      │
            │ Precondition │  output = c_skip · noisy_input + c_out · F(x)
            └──────┬───────┘
                   │
                   ▼
            Denoised frame
              (3, 84, 84)
```


## ResBlock Detail

Each ResBlock has two conv layers, each preceded by AdaGroupNorm:

```
input x ──────────────────────────────────┐ (residual/skip)
    │                                     │
    ▼                                     │
┌───────────────┐                         │
│ AdaGroupNorm  │◄── cond vector          │
│               │    scale, shift =       │
│ normalize x,  │    Linear(cond)         │
│ then scale/   │                         │
│ shift by cond │    x = norm(x)*(1+s)+b  │
└───────┬───────┘                         │
        │                                 │
        ▼                                 │
    SiLU activation                       │
        │                                 │
        ▼                                 │
   Conv2d 3×3                             │
        │                                 │
        ▼                                 │
┌───────────────┐                         │
│ AdaGroupNorm  │◄── cond vector          │
└───────┬───────┘                         │
        │                                 │
        ▼                                 │
    SiLU activation                       │
        │                                 │
        ▼                                 │
   Conv2d 3×3 (zero-init)                 │
        │                                 │
        ▼                                 │
      (+) ◄───────────────────────────────┘
        │
        ▼
    output = x + residual
```


## Inference: 3-Step Euler Denoising

At inference we don't have a target frame. We start from pure noise
and denoise in 3 steps:

```
Step 0          Step 1          Step 2          Final
σ = 5.0         σ = 0.54        σ = 0.06        σ = 0

┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│░░░░░░░░░│     │▓▒░ ▒▓▒░ │     │▓█▒▓█▓▒▓ │     │████████ │
│░░░░░░░░░│ ──► │▒▓░▒▓▒░▒ │ ──► │█▓▒█▓█▒█ │ ──► │████████ │
│░░░░░░░░░│     │░▒▓░▒▓░▒ │     │▓█▓▒█▓█▓ │     │████████ │
│ noise   │     │ structure│     │ details  │     │ clean   │
└─────────┘     └─────────┘     └─────────┘     └─────────┘

Each step:
  denoised = model(noisy_x, context, action, σ)
  d = (noisy_x - denoised) / σ        ← estimated "direction to clean"
  x_next = noisy_x + (σ_next - σ) · d ← step toward clean
```


## Why Fourier Features for σ (not raw scalar)

If we pass σ directly to a Linear layer, the network can only learn:

```
output = w · σ + b          ← a straight line
```

But the network needs to do very different things at different noise levels:

```
Network behavior needed:

  "hallucinate         "refine         "tiny
   everything"          edges"          corrections"
      ▲                   ▲                ▲
      │    ╱               │               │
      │   ╱                │    ╱╲         │         ╱
      │  ╱                 │   ╱  ╲        │        ╱
      │ ╱                  │  ╱    ╲       │       ╱
      │╱                   │ ╱      ╲      │      ╱
  ────┼──────── σ      ────┼──────── σ  ───┼──────── σ
      0    5               0    5          0    5

  Linear can do this.   Can't do this.   Can't do this.
  But real behavior needs all of these simultaneously.
```

Fourier features create a **basis of wiggly functions**:

```
σ ──► [ sin(2π·f₁·σ),  cos(2π·f₁·σ),     ← slow oscillation
        sin(2π·f₂·σ),  cos(2π·f₂·σ),     ← medium oscillation
        sin(2π·f₃·σ),  cos(2π·f₃·σ),     ← fast oscillation
        ...                                   (64 dimensions total)
      ]
```

Now a Linear layer on top of these can combine wiggles at different
frequencies to approximate **any** function of σ:

```
output = w₁·sin(f₁·σ) + w₂·cos(f₁·σ) + w₃·sin(f₂·σ) + ...

This is literally a Fourier series — it can approximate any shape.
```

Same idea as positional encoding in transformers: turn a single
number into a rich vector so linear layers can learn nonlinear
functions of it.


## Model Sizes

| Size   | Channels            | Params | Use case                |
|--------|---------------------|--------|-------------------------|
| tiny   | [32, 32, 64, 64]    | ~0.5M  | Smoke tests             |
| small  | [64, 64, 64, 64]    | ~3M    | DIAMOND Atari baseline  |
| medium | [64, 128, 128, 128] | ~12M   | Quake III starting point|
| large  | [128, 128, 256, 256]| ~35M   | If medium underfits     |
