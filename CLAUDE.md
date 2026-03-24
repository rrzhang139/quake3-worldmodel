## Communication Style

**Show the execution flow, not abstractions.** When working on code or experiments:

1. **After EVERY code edit**: Explain what changed, why, and how it connects to the full data flow. Trace the path: input data → preprocessing → model forward pass → loss → optimizer step → output. Don't say "updated the training loop" — say "changed line 358 in model.py where sigma is sampled: was `exp(randn)*0.5` (uniform-ish), now `exp(randn*1.2 - 0.4)` (DIAMOND's log-normal, peaks around sigma=0.67 which means most training focuses on medium noise levels where the model learns the most)."

2. **Hyperparameter changes need intuition + example**: Don't just say "set weight_decay=1e-2". Say: "AdamW weight_decay=1e-2 means each step, weights shrink by 1%. A weight of 0.5 becomes 0.495. This prevents any single weight from growing too large, which regularizes the model — important when we have 20M params but only 1M training frames."

3. **Show the linear sequential flow**: When describing architecture or training, walk through it step by step like a debugger:
   - "Frame comes in as uint8 (120, 160, 3) → Episode.load() divides by 255, multiplies by 2, subtracts 1 → now float32 in [-1,1] → dataset returns context (4, 3, 120, 160) + target (3, 120, 160) + action (scalar int) → model.training_loss() samples sigma from log-normal → adds noise to target → concatenates noisy_target with context along channel dim → U-Net predicts clean target → EDM loss weights by sigma → backward → AdamW step"

4. **Frequent status updates**: After each tool call, briefly say what happened and what's next. Don't batch 5 edits silently then summarize.

5. **Don't skip details**: If there are 10 lines of code that matter, show all 10 and explain each. The user wants holistic understanding, not summaries.

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

## Artifact Storage Rules

**NEVER store large files in git.** Use the right storage for each type:

| Type | Where | Naming | Example |
|------|-------|--------|---------|
| Model checkpoints (.pt) | **W&B Artifacts** | `quake3-wm-<description>` | `quake3-wm-run8-160x120-4ep` |
| Datasets (episodes) | **HuggingFace** | `rzhang139/vizdoom-episodes/<name>` | `episodes_160x120_5k` |
| IDM weights | **W&B Artifacts** | `quake3-idm` | `quake3-idm` |
| Eval videos | **W&B Media** | logged via `wandb.Video()` | attached to eval runs |
| Tweet media (small GIFs <2MB) | **Git** (assets/tweets/) | OK in git, small files | `tweet1_media.gif` |

**Git repo should stay small.** Only code, configs, markdown, and small images (<2MB) go in git. Everything else goes to W&B or HuggingFace.

**Download checkpoint on pod:**
```python
import wandb, shutil
wandb.login(key=os.environ["WANDB_API_KEY"])
art = wandb.Api().artifact("rzhang139/quake3-worldmodel/quake3-wm-run8-160x120-4ep:latest")
art.download("/workspace/ckpt")
shutil.copy("/workspace/ckpt/best.pt", "best.pt")
```

## Model Sizes

| Size   | Channels            | Params | VRAM @ batch=128 (latent) | Right-size GPU          |
|--------|---------------------|--------|---------------------------|-------------------------|
| tiny   | [32, 32, 64, 64]    | ~0.5M  | ~0.5 GB                   | Any GPU (even free tier)|
| small  | [64, 64, 64, 64]    | ~3M    | ~0.8 GB                   | RTX 3070 ($0.07/hr)     |
| medium | [64, 128, 128, 128] | ~7.6M  | ~1.6 GB                   | RTX A4000 ($0.09/hr)    |
| large  | [128, 128, 256, 256]| ~35M   | ~5–8 GB                   | RTX 3090 ($0.11/hr)     |

**GPU selection rule**: estimate VRAM first, then pick cheapest GPU with 2x headroom.
- VRAM ≈ params×4 bytes (weights) + params×8 bytes (Adam states) + batch activations
- For latent models (4ch, 15×20): activations ≈ batch × 0.05 MB — negligible
- Don't use A40/A100 unless model is >10GB VRAM; they cost 3–5x more than RTX 3090
- Always use `cloudType: ALL` for overnight runs (avoids community preemption) — only upgrade to SECURE if ALL fails repeatedly

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
ssh -tt -i ~/.ssh/runpod <podHostId>@ssh.runpod.io << 'SSHEOF'
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
- **CRITICAL: SSH address is `<podHostId>@ssh.runpod.io`** — NOT `<pod_id>@ssh.runpod.io`. Get it from:
  ```bash
  curl -s "https://api.runpod.io/graphql?api_key=$KEY" -H "Content-Type: application/json" \
    -d '{"query": "{ pod(input: { podId: \"POD_ID\" }) { machine { podHostId } } }"}'
  # Result: "podHostId": "cyzx60pwcdht7t-64411f5f" → ssh cyzx60pwcdht7t-64411f5f@ssh.runpod.io
  ```
- **tmux not always available**: `apt-get install tmux` requires `apt-get update` first; use `nohup ... &` as fallback
- **Direct IP SSH (`213.x.x.x:port`) does NOT work** — always use the gateway `ssh.runpod.io`
- **`dockerArgs` breaks SSH** — overrides container startup, preventing SSH daemon from running. Never use `dockerArgs` if you need SSH.

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
