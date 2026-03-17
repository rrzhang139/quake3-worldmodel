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
            Before finishing, ensure ALL of the following are done:
            1. Checkpoint uploaded to W&B (artifact name: quake3-wm-<description>)
            2. Eval metrics logged (teacher-forced PSNR, AR PSNR, copy baseline, PSNR curve)
            3. Pod stopped or auto-shutdown scheduled
            4. Progress log updated at memory/project_progress.md
            5. Any new learnings saved to memory (gotchas, infra specs, what worked/didn't)
            Return {"ok": true} if all done, {"ok": false, "reason": "what's missing"} if not.
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

## Training
- Use `nohup` for long runs (SSH disconnects kill processes)
- Set up timed auto-shutdown for overnight runs: `sleep <seconds> && pkill && upload && stop pod`
- Monitor via RunPod API GPU util, NOT just SSH
- Log to W&B: `--wandb`

## Eval (MANDATORY before stopping)
Run comprehensive eval with `src/eval.py`:
- Teacher-forced PSNR (single-step, real context)
- AR rollout PSNR (32-frame, 8-frame)
- Copy-baseline comparison
- PSNR-by-step curve
- FVD
Key question: does teacher-forced PSNR beat copy baseline?

## Post-experiment (exit checklist)
1. Upload `best.pt` to W&B: `quake3-wm-<run_description>`
2. Stop/terminate pod
3. Update `memory/project_progress.md` with results
4. Update `memory/reference_infra_specs.md` if new specs discovered
5. Note what worked and what didn't for next run

## Arguments
$ARGUMENTS

## Current best results (for comparison)
- 84x84: teacher-forced 21.4 dB, AR 19.2 dB (Run 7, large 20.6M, 10K eps)
- 160x120: teacher-forced 19.3 dB, AR 17.4 dB (Run 9, large 20.6M, 5K eps, 1 epoch — undertrained)
- Copy baseline: ~19-21 dB depending on resolution
- Action ratio: 8.5x at cfg=1.5 (Run 7)
