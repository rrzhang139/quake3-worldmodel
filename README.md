# Quake III Arena — World Model

**Goal:** Build the first world model for Quake III Arena. Given the current frame + player action, predict the next frame. Chain predictions → playable "neural Quake III."

**Status:** Scoping & infrastructure setup

## Why Quake III?

- **Nobody has done it.** DOOM has GameNGen, Quake II has WHAMM, CS:GO has DIAMOND. Quake III is untouched.
- **Best infrastructure of any untouched FPS.** DeepMind Lab (built on ioquake3) provides a production-quality gym environment. OpenArena gives us GPL assets.
- **Unlimited free data.** Run RL agents in DeepMind Lab → millions of frame+action pairs generated on CPU.
- **Rich 3D game.** True polygonal 3D (not DOOM's 2.5D raycasting), curved surfaces, dynamic lighting, fast-paced deathmatch.

## The Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Data Gen     │     │ Train        │     │ Evaluate     │     │ Demo         │
│ (CPU, local) │ ──▶ │ (GPU, RunPod)│ ──▶ │ (GPU)        │ ──▶ │ (interactive)│
│              │     │              │     │              │     │              │
│ DeepMind Lab │     │ Fork DIAMOND │     │ PSNR/LPIPS   │     │ FastAPI +    │
│ + RL agents  │     │ diffusion    │     │ FVD          │     │ WebSocket    │
│ → frame+act  │     │ U-Net        │     │ IDM F1       │     │ browser play │
│   .npz files │     │ on RunPod    │     │ side-by-side │     │              │
└──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
```

## Quick Start

```bash
cd quake3-worldmodel && bash setup_env.sh
# TODO: data collection, training, eval scripts
```

## Key References

| Paper | Game | Architecture | Params | Our Relevance |
|-------|------|-------------|--------|---------------|
| [DIAMOND](https://github.com/eloialonso/diamond) (NeurIPS 2024) | Atari / CS:GO | Diffusion U-Net | 4.4M-381M | **Fork target** |
| [GameNGen](https://gamengen.github.io/) (Google 2024) | DOOM | Fine-tuned SD 1.4 | 860M | Noise augmentation technique |
| [WHAMM](https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/) (Microsoft 2025) | Quake II | MaskGIT | 750M | Same franchise, different engine |
| [DeepMind FTW](https://arxiv.org/abs/1807.01281) (2018) | Quake III CTF | RL agent | — | Data generation approach |
| [Diffusion Forcing](https://arxiv.org/abs/2407.01392) (2024) | DMLab mazes | Diffusion | — | Tested on Q3 engine (simplified) |

## Related Work on Quake III Engine

These models used DeepMind Lab (Quake III engine) but with **simplified procedural mazes**, not actual Q3 Arena gameplay:

- **DreamerV3** — latent-space world model, tested on 30 DMLab tasks
- **TECO** — 300-frame video prediction in DMLab mazes at 64x64
- **Diffusion Forcing** — 2000+ frame rollouts on DMLab

We are the first to target **actual Quake III Arena** maps, textures, and deathmatch gameplay.
