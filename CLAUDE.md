## Shared Infrastructure
This project is part of a multi-repo research setup. Shared GPU/SSH/RunPod infra lives in the
sibling `personal-research/` repo (locally at `../personal-research/`, on pods at `/workspace/code/personal-research/`).

- **Provider docs**: `../personal-research/providers/` (RunPod setup, pricing, SSH patterns)
- **RunPod scripts**: `../personal-research/runpod/` (setup.sh, restart.sh, save.sh, offboard.sh)
- **Root CLAUDE.md**: `../personal-research/CLAUDE.md` (GPU cost philosophy, SSH conventions, W&B artifact mandate)

**Rules**:
- If you discover a shared infra improvement (RunPod gotcha, SSH pattern, W&B convention), push it to `../personal-research/CLAUDE.md`
- Project-specific infra (vizdoom install, EDM details) stays in THIS repo's CLAUDE.md

## Autonomy & Infrastructure Permissions
- **DO NOT ask for permission** on routine infrastructure operations: starting/stopping/creating pods, transferring data, uploading to W&B, collecting data, running evals. Just do it.
- **DO ask** before: spending >$10 in a single action, deleting data/weights that aren't backed up, or sending tweets/public messages.
- **API keys** are in `../personal-research/runpod/.env` — read that file for RunPod, W&B, GitHub, HF tokens. X/Twitter keys are in this repo's `.env`.
- **RunPod API**: Use `https://api.runpod.io/graphql?api_key=<RUNPOD_API_KEY>` for pod management. If a pod can't start (host full), create a new one. Collect data on-pod rather than transferring large datasets.
- **W&B mandate**: Always upload checkpoints to W&B before stopping a pod. Artifact naming: `quake3-wm-<description>`.

# Quake III World Model — Project Instructions

## Overview

Building the first world model for Quake III Arena. EDM diffusion U-Net (inspired by DIAMOND), trained on ViZDoom data, deployed on RunPod.

## Key Architecture Decisions

- **Data source**: ViZDoom (DOOM engine) for prototyping. DeepMind Lab is effectively dead (last commit Jan 2023, requires Bazel, Linux-only, no pip package). Swap to DMLab on RunPod later if needed.
- **Model**: Custom EDM diffusion U-Net inspired by [DIAMOND](https://github.com/eloialonso/diamond) (MIT license). ~0.5-35M params depending on size.
- **Action conditioning**: Adaptive group normalization (AdaGroupNorm) — action embedding + noise level encoding combined into conditioning vector, injected at every ResBlock
- **Resolution**: 84x84 (ViZDoom native 160x120, resized via PIL)
- **Training hardware**: 1x RTX 3090/4090/A5000 on RunPod ($0.11-0.39/hr)
- **Diffusion**: EDM preconditioning (Karras et al.), NOT DDPM. 3-step Euler ODE sampling at inference.

## Project Structure

```
quake3-worldmodel/
├── CLAUDE.md           # This file
├── README.md           # Project overview
├── SCOPE.md            # Full scope document
├── ARCHITECTURE.md     # Detailed ASCII architecture diagrams
├── setup_env.sh        # Environment setup (auto-detects GPU)
├── requirements.txt    # Python dependencies
├── .gitignore          # Ignores .venv/, data/, experiments/, wandb/, *.pt, *.tar.gz
├── data/
│   └── episodes/       # Collected episode .pt files
└── src/
    ├── episode.py      # Episode dataclass (obs/act/rew/end/trunc)
    ├── collect.py      # ViZDoom data collection (10 discrete actions)
    ├── dataset.py      # PyTorch dataset — samples (context, action, target) tuples
    ├── model.py        # EDM diffusion U-Net with AdaGroupNorm
    ├── train.py        # Training loop with W&B, checkpointing, sample generation
    └── eval.py         # Autoregressive rollout evaluation (PSNR, videos, strips)
```

## Episode Format

```python
Episode:
  obs: FloatTensor  (T+1, C, H, W) in [-1, 1]   # T+1 because we need target frame
  act: LongTensor   (T,)                          # discrete action indices 0-9
  rew: FloatTensor  (T,)
  end: BoolTensor   (T,)
  trunc: BoolTensor (T,)
```

Saved as `.pt` files with uint8 obs on disk (saves 4x space). Converted to float [-1,1] on load.

## Datasets

All datasets stored on HuggingFace: `rzhang139/vizdoom-episodes`

| Dataset | Resolution | Episodes | Frames | Size |
|---------|-----------|----------|--------|------|
| `episodes_160x120_5k` | 160x120 (native) | 5,000 | ~1M | ~19GB |
| `episodes_84x84_10k` | 84x84 (resized) | 10,000 | ~2M | ~40GB |

**Download on pod:**
```bash
pip install huggingface_hub
huggingface-cli download rzhang139/vizdoom-episodes \
    --repo-type dataset \
    --include "episodes_160x120_5k/*" \
    --local-dir data \
    --token $HF_TOKEN
```

**Collect new data:**
```bash
# Native 160x120 (recommended — matches GameNGen resolution)
python src/collect.py --num_episodes 5000 --output data/episodes_160x120 --max_steps 300 --policy mixed --res 0 --screen_res 160x120

# 84x84 (legacy)
python src/collect.py --num_episodes 10000 --output data/episodes_84x84 --max_steps 300 --policy mixed --res 84
```

**Always download from HF instead of re-collecting on pods.** Collection is CPU-bound and slow on pod CPUs.

## Model Sizes

| Size   | Channels            | Params | Use case                |
|--------|---------------------|--------|-------------------------|
| tiny   | [32, 32, 64, 64]    | ~0.5M  | Smoke tests             |
| small  | [64, 64, 64, 64]    | ~3M    | Baseline training       |
| medium | [64, 128, 128, 128] | ~12M   | Production              |
| large  | [128, 128, 256, 256]| ~35M   | If medium underfits     |

## Dependencies

- `vizdoom` — DOOM environment for data collection
- `torch`, `torchvision` — PyTorch
- `wandb` — experiment tracking
- `numpy`, `pillow` — data processing
- `imageio` — video generation for eval

## W&B

- **Project**: `rzhang139/quake3-worldmodel`
- **Entity**: `rzhang139`
- Log: training loss, PSNR, rollout videos, side-by-side comparisons
- **ALWAYS upload checkpoints to W&B Artifacts** after training. Never stop a pod without uploading best.pt/final.pt first. Naming: `quake3-wm-<run_description>` type `model`.

## Data Collection

```bash
# Collect 1000 episodes locally (~40s, ~3.8GB)
cd quake3-worldmodel
source .venv/bin/activate
python src/collect.py --num_episodes 1000 --output data/episodes --max_steps 300
```

- Achieves ~5K fps on CPU (no GPU needed)
- 10 discrete actions (ACTIONS_V2): movement + turning + strafing + shooting
- Uses ViZDoom deathmatch scenario
- Data is CPU-bound, collect locally on Mac

## Training

```bash
# Local smoke test (tiny model, 5 episodes)
python src/train.py --data data/episodes --epochs 5 --model_size tiny --batch_size 16

# RunPod production run (small model, 1000 episodes)
python -u src/train.py --data data/episodes --epochs 100 --batch_size 32 \
    --model_size small --lr 1e-4 --noise_aug 0.1 --wandb \
    --output experiments/run
```

Always use `python -u` for unbuffered output when redirecting to log files.

## Evaluation

```bash
python src/eval.py --checkpoint experiments/run/best.pt --data data/episodes \
    --output experiments/run/eval --model_size small --wandb
```

Generates: PSNR metrics, side-by-side image strips, comparison MP4 videos.

## RunPod Setup

**IMPORTANT**: Always follow the existing RunPod infrastructure in `../personal-research/runpod/` (setup.sh, restart.sh, save.sh).

### Pod creation
```bash
# Use runpodctl or web UI
# RTX 3090 spot ($0.11/hr) for testing, RTX 4090 ($0.20-0.39/hr) for production
# Template: RunPod PyTorch 2.x (has CUDA pre-installed)
# Volume: 50GB persistent at /workspace/
```

### After pod creation
```bash
# 1. Upload .env (scp DOES NOT WORK through RunPod SSH gateway)
#    Write .env content via heredoc instead:
ssh -tt -i ~/.ssh/runpod <SSH_ADDRESS> << 'SSHEOF'
cat > /workspace/.env << 'ENVEOF'
HF_TOKEN=...
WANDB_API_KEY=...
GITHUB_TOKEN=...
ENVEOF
exit
SSHEOF

# 2. Run setup following ../personal-research/runpod/setup.sh pattern
# 3. Clone repo, create venv, install deps
# 4. Collect data or transfer from local
# 5. Launch training in tmux (NOT nohup — tmux survives SSH disconnect AND you can reattach)
```

### Data transfer to pod
Since scp doesn't work through RunPod gateway, options:
1. **Git push data** (if small enough)
2. **Collect on pod** — ViZDoom works on Linux, ~40s for 1000 episodes
3. **tar.gz + base64 heredoc** (for <100MB)
4. **Cloud storage** (S3/GCS presigned URL)

## Development Philosophy

**Breadth-first, not depth-first.** Get each part working end-to-end at minimum viable quality, then iterate. Don't perfect the dataloader before seeing a single generated frame.

## Gotchas / Lessons Learned

### DeepMind Lab is dead
- Last commit Jan 2023, no pip package, requires Bazel build, Linux-only
- Doesn't work on Mac, finicky even on Linux
- Use ViZDoom for prototyping instead — pip-installable, cross-platform

### Corporate pip proxy
- If `UV_INDEX_URL`, `UV_DEFAULT_INDEX`, `UV_EXTRA_INDEX_URL` env vars are set (e.g., Zoom Artifactory), they intercept all pip/uv installs
- Fix: `unset UV_INDEX_URL UV_DEFAULT_INDEX UV_EXTRA_INDEX_URL` before installing
- Also check `~/.config/pip/pip.conf` for corporate index URLs

### PyTorch checkpoint serialization
- `torch.save` with numpy scalar values (e.g., `np.float64` loss) causes `weights_only=True` to fail on load
- Fix: always cast to Python float before saving: `float(loss)`
- For own checkpoints in eval.py, use `weights_only=False`

### RunPod SSH gateway quirks
- Requires `-tt` flag for PTY allocation
- Ignores command arguments — must use heredoc (`<< 'SSHEOF'`)
- `scp` / `sftp` don't work (`subsystem request failed on channel 0`)
- Output is very noisy (MOTD, PTY control characters)
- tmux doesn't inherit parent env vars — must `source .bashrc_pod` inside tmux

### EDM diffusion (not DDPM)
- Critical for stable autoregressive generation (long rollouts)
- Uses c_skip/c_out/c_in/c_noise preconditioning weights
- Log-normal noise sampling during training
- 3-step Euler ODE solver at inference (sigma: 5.0 → 0.54 → 0.06 → 0)

### Always maximize GPU utilization
- Before launching training, run a quick batch size sweep to find the largest batch that fits in VRAM (~80% target)
- Use linear scaling rule for LR: if batch 4x larger, LR should be ~4x larger
- On RTX A5000 (24GB), small model: batch=128 uses 12.7GB (good), batch=256 OOMs
- On RTX 4090 (24GB), expect similar limits
- This can easily give 3-4x speedup vs a conservative default batch size

### GameNGen noise augmentation
- Adding small noise (sigma=0.1) to context frames during training
- Prevents autoregressive drift at inference (model trained on clean context but receives own noisy predictions)
- Controlled via `--noise_aug` flag in train.py

### ViZDoom on Mac
- `pip install vizdoom` works but must use venv's pip, not conda's
- If `which pip` points to conda, vizdoom installs into wrong Python
- Always use `uv pip install` inside activated venv

### ViZDoom not in RunPod pytorch image
- Fresh `runpod/pytorch` containers don't have vizdoom. Must `uv pip install vizdoom` after `setup_env.sh`.
- Collect data on pod (~50s for 1000 episodes) rather than transferring ~4GB.

### Fresh pods lack tmux
- `apt-get install tmux` is slow on fresh containers. Fallback: `nohup python -u ... > /workspace/train.log 2>&1 &` then check logs in a separate SSH call.
- Never pipe long-running commands through `tail -N` in a heredoc — it buffers and hangs the SSH session.

### Action conditioning fails on random policy data
- Random agent spins aimlessly — model learns to ignore actions (action_test ratio ~1.1x).
- Use `--policy mixed` (70% scripted / 30% random) for coherent trajectories.
- Test with `src/action_test.py` — ratio > 2x = PASS.

## Research Notes

See `RESEARCH.md` for detailed experiment logs and hypotheses (DoomMWM multiplayer, single-view curriculum, etc.)

## Key References

- DIAMOND code: https://github.com/eloialonso/diamond
- GameNGen (noise augmentation): https://arxiv.org/abs/2408.14837
- WHAMM (Quake II world model): https://www.microsoft.com/en-us/research/articles/whamm-real-time-world-modelling-of-interactive-environments/
- EDM (Karras et al.): https://arxiv.org/abs/2206.00364
- ViZDoom: https://vizdoom.farama.org/

## Cost Tracking

| Date       | Provider | GPU      | Hours | Cost  | Purpose                    |
|------------|----------|----------|-------|-------|----------------------------|
| 2026-03-12 | RunPod   | RTX A5000| ~14h  | ~$3.10| Single-player baseline (small, 100ep, random policy, loss→0.032) |
| 2026-03-12 | RunPod   | RTX 3090 | ~17h  | ~$4   | Mixed policy retrain (small, 100ep, action conditioning fix) |
