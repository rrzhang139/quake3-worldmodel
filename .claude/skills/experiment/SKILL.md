---
name: experiment
description: Run a training experiment end-to-end. Handles data loading, training, eval, W&B upload, and pod shutdown. Use for any new training run.
argument-hint: "[description of what to train, e.g. 'large model 160x120 noise_aug=0.5']"
user-invocable: true
hooks:
  Stop:
    - hooks:
        - type: prompt
          prompt: |
            Experiment exit checklist (skip items that don't apply):
            1. Checkpoint uploaded to W&B? (artifact: quake3-wm-<description>)
            2. Eval run with ALL metrics? (PSNR, SSIM, LPIPS, FVD, action accuracy, world consistency)
            3. Pod stopped or auto-shutdown scheduled?
            4. Progress log updated? (.claude/skills/experiment/experiment_log.md AND memory/project_progress.md)
            5. Learnings saved? (memory files for gotchas, infra specs)
            Return {"ok": true} — this is advisory.
---

# Experiment Skill

You are running a training experiment for the Quake III world model. Follow this checklist rigorously.

## Pre-flight
- Read `memory/project_progress.md` for context on previous runs
- Read `memory/reference_infra_specs.md` to pick correct pod specs
- Read `memory/reference_datasets.md` to find existing datasets (DON'T re-collect)
- Decide: model size, resolution, batch size, noise_aug, epochs, data

## Data
- **Always download from HuggingFace first**: `rzhang139/vizdoom-episodes`
- Available: `episodes_160x120_5k` (native res), `episodes_84x84_10k`
- Only collect new data if resolution/policy needs to change
- Use `--data_mode ram` for training (NOT streaming)

## Pod Setup Gotchas (things that broke before)
- **Always activate venv**: `source /workspace/quake3-worldmodel/.venv/bin/activate` (or wherever venv is). System python often lacks wandb, lpips, etc.
- **Find data path before running eval**: `find /workspace -maxdepth 3 -name "episode_000000.pt"` — data may be at different paths on different pods
- **Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True** for 160x120 runs
- **Use on-demand (SECURE cloud) for runs >2h** — community pods get preempted

## Training
- Use `nohup` for long runs (SSH disconnects kill processes)
- Set up timed auto-shutdown for overnight runs: `sleep <seconds> && pkill && upload && stop pod`
- For auto-shutdown: ALWAYS `source venv/bin/activate` before running W&B upload in the shutdown script
- Monitor via RunPod API GPU util, NOT just SSH
- Log to W&B: `--wandb`

## Eval (MANDATORY before stopping)
Run comprehensive eval with `src/eval.py` — includes ALL metrics:
- Teacher-forced PSNR + SSIM (single-step, real context)
- AR rollout PSNR + SSIM (32-frame, 8-frame)
- Copy-baseline comparison + delta
- PSNR-by-step curve (drift analysis)
- FVD (temporal coherence)
- Action accuracy via IDM (if --idm_checkpoint provided)
- World consistency (turn left+right test)

Key questions eval answers:
1. Does teacher-forced PSNR beat copy baseline? (is model learning?)
2. Where does drift kick in? (PSNR-by-step curve)
3. Do actions work correctly? (IDM accuracy)
4. Is the world spatially consistent? (turn test)

## W&B Upload
CRITICAL: Always activate venv before W&B operations:
```bash
source /workspace/quake3-worldmodel/.venv/bin/activate  # or wherever venv is
export WANDB_API_KEY=<key>
python -c "import wandb; wandb.login(key=...); ..."
```
Artifact naming: `quake3-wm-<description>` (e.g. `quake3-wm-run8-160x120-4ep`)

## Post-experiment (exit checklist)
1. Upload `best.pt` to W&B: `quake3-wm-<run_description>`
2. Stop/terminate pod
3. Update `.claude/skills/experiment/experiment_log.md` with run results
4. Update `memory/project_progress.md` with summary
5. Update `memory/reference_infra_specs.md` if new specs discovered
6. Note what worked and what didn't

## Arguments
$ARGUMENTS

## Current best results (for comparison)
- 84x84: teacher-forced 21.4 dB, SSIM ~0.27, AR 19.2 dB (Run 7, large 20.6M, 10K eps)
- 160x120 (1 epoch): teacher-forced 19.3 dB, SSIM 0.34, AR 17.4 dB (Run 9)
- 160x120 (4 epochs): loss=0.042, eval pending (Run 8)
- Copy baseline: ~19-21 dB depending on resolution
- Action ratio: 8.5x at cfg=1.5 (Run 7)
