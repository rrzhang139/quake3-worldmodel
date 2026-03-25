"""Evaluate the latent-space world model.

Data flow:
  episode obs (T+1, 3, 120, 160) uint8
    → float [-1,1]
    → SD VAE encoder → (T+1, 4, 15, 20) latents (scaled by 0.18215)
    → model.sample(context_latents, action) → predicted next latent
    → SD VAE decoder → (3, 120, 160) float [-1,1] → uint8
    → compare with ground truth pixel frame → PSNR

Metrics:
  1. Teacher-forced PSNR (real context → predict 1 frame)
  2. Autoregressive PSNR curve (frames 1,2,4,8,16,32)
  3. Copy-baseline PSNR (always predict previous frame)

Usage:
    python src/eval_latent.py --checkpoint /tmp/wm-latent-medium-ckpt/best.pt \
        --data data/episodes_160x120_5k --num_episodes 20 --wandb
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL

from model import make_denoiser


VAE_MODEL = "stabilityai/sd-vae-ft-mse"
SCALE_FACTOR = 0.18215
NUM_CONTEXT = 4


# ─── VAE helpers ──────────────────────────────────────────────────────────────

def encode_frames(vae, frames_uint8: torch.Tensor, device: str, batch_size: int = 16) -> torch.Tensor:
    """(T, 3, H, W) uint8 → (T, 4, H/8, W/8) scaled latents."""
    frames_f = frames_uint8.float().div(255).mul(2).sub(1)
    latents = []
    for start in range(0, len(frames_f), batch_size):
        batch = frames_f[start:start + batch_size].to(device)
        with torch.no_grad():
            post = vae.encode(batch)
            z = post.latent_dist.mode() * SCALE_FACTOR
        latents.append(z.cpu())
    return torch.cat(latents, dim=0)


def decode_latents(vae, latents: torch.Tensor, device: str, batch_size: int = 16) -> torch.Tensor:
    """(T, 4, H/8, W/8) scaled latents → (T, 3, H, W) uint8."""
    pixels = []
    for start in range(0, len(latents), batch_size):
        batch = latents[start:start + batch_size].to(device)
        with torch.no_grad():
            out = vae.decode(batch / SCALE_FACTOR).sample  # (B, 3, H, W)
        out = out.clamp(-1, 1).add(1).div(2).mul(255).byte().cpu()
        pixels.append(out)
    return torch.cat(pixels, dim=0)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """PSNR between two (C, H, W) uint8 tensors."""
    mse = ((a.float() - b.float()) ** 2).mean().item()
    if mse == 0:
        return float("inf")
    return 10 * np.log10(255 ** 2 / mse)


# ─── Eval loops ───────────────────────────────────────────────────────────────

def eval_teacher_forced(model, vae, episodes, device, cfg_scale=1.5):
    """Single-step PSNR: given real context → predict 1 frame → compare."""
    psnrs, copy_psnrs = [], []
    for ep in episodes:
        obs = ep["obs"]          # (T+1, 3, H, W) uint8
        act = ep["act"]          # (T,) int
        T = len(act)
        if T < NUM_CONTEXT + 1:
            continue
        latents = encode_frames(vae, obs, device)  # (T+1, 4, 15, 20)

        for t in range(NUM_CONTEXT, min(T, NUM_CONTEXT + 30)):
            ctx_lat = latents[t - NUM_CONTEXT:t].unsqueeze(0).to(device)  # (1, 4, 4, 15, 20)
            a = torch.tensor([act[t].item()], device=device)
            with torch.no_grad():
                pred_lat = model.sample(ctx_lat, a, num_steps=3, cfg_scale=cfg_scale)  # (1, 4, 15, 20)
            pred_px = decode_latents(vae, pred_lat, device)  # (1, 3, 120, 160)
            gt_px = obs[t + 1]  # (3, 120, 160) uint8
            psnrs.append(psnr(pred_px[0], gt_px))
            copy_psnrs.append(psnr(obs[t], gt_px))

    return float(np.mean(psnrs)), float(np.mean(copy_psnrs))


def eval_autoregressive(model, vae, episodes, device, rollout_len=32, cfg_scale=1.5):
    """Autoregressive PSNR curve: given 4 seed frames, roll out for rollout_len steps."""
    checkpoints = [1, 2, 4, 8, 16, 32]
    checkpoints = [c for c in checkpoints if c <= rollout_len]
    all_psnrs = {c: [] for c in checkpoints}
    all_copy_psnrs = {c: [] for c in checkpoints}

    for ep in episodes:
        obs = ep["obs"]      # (T+1, 3, H, W) uint8
        act = ep["act"]      # (T,) int
        T = len(act)
        if T < NUM_CONTEXT + rollout_len:
            continue
        latents = encode_frames(vae, obs, device)  # (T+1, 4, 15, 20)

        # Seed context: frames [0..NUM_CONTEXT-1]
        ctx_buf = list(latents[:NUM_CONTEXT])  # list of (4, 15, 20) latents

        for step in range(rollout_len):
            t = NUM_CONTEXT + step
            ctx_lat = torch.stack(ctx_buf[-NUM_CONTEXT:], dim=0).unsqueeze(0).to(device)
            a = torch.tensor([act[t].item()], device=device)
            with torch.no_grad():
                pred_lat = model.sample(ctx_lat, a, num_steps=3, cfg_scale=cfg_scale)  # (1, 4, 15, 20)
            ctx_buf.append(pred_lat[0].cpu())

            step_idx = step + 1
            if step_idx in checkpoints:
                pred_px = decode_latents(vae, pred_lat, device)[0]  # (3, H, W) uint8
                gt_px = obs[t + 1]
                all_psnrs[step_idx].append(psnr(pred_px, gt_px))
                all_copy_psnrs[step_idx].append(psnr(obs[t], gt_px))

    return (
        {c: float(np.mean(v)) for c, v in all_psnrs.items() if v},
        {c: float(np.mean(v)) for c, v in all_copy_psnrs.items() if v},
    )


def make_comparison_video(model, vae, ep, device, rollout_len=64, cfg_scale=1.5):
    """Returns (rollout_len, H, 2*W, 3) uint8 numpy array: GT | pred side-by-side."""
    obs = ep["obs"]
    act = ep["act"]
    T = len(act)
    if T < NUM_CONTEXT + rollout_len:
        rollout_len = T - NUM_CONTEXT

    latents = encode_frames(vae, obs, device)
    ctx_buf = list(latents[:NUM_CONTEXT])
    frames = []

    for step in range(rollout_len):
        t = NUM_CONTEXT + step
        ctx_lat = torch.stack(ctx_buf[-NUM_CONTEXT:], dim=0).unsqueeze(0).to(device)
        a = torch.tensor([act[t].item()], device=device)
        with torch.no_grad():
            pred_lat = model.sample(ctx_lat, a, num_steps=3, cfg_scale=cfg_scale)
        ctx_buf.append(pred_lat[0].cpu())

        pred_px = decode_latents(vae, pred_lat, device)[0]  # (3, H, W) uint8
        gt_px = obs[t + 1]  # (3, H, W) uint8

        gt_np = gt_px.permute(1, 2, 0).numpy()
        pred_np = pred_px.permute(1, 2, 0).numpy()
        frames.append(np.concatenate([gt_np, pred_np], axis=1))

    return np.stack(frames, axis=0)  # (T, H, 2W, 3)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", default="data/episodes_160x120_5k")
    parser.add_argument("--num_episodes", type=int, default=20)
    parser.add_argument("--rollout_len", type=int, default=32)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--model_size", default="medium")
    parser.add_argument("--num_videos", type=int, default=3)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output", default="experiments/eval_latent_medium")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = make_denoiser(
        num_actions=10,
        img_size=20,
        img_channels=4,
        num_context_frames=NUM_CONTEXT,
        model_size=args.model_size,
        cfg_drop_prob=0.15,
        action_aux_weight=0.1,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded: epoch={ckpt['epoch']+1}, loss={ckpt['loss']:.4f}")

    # Load VAE
    print(f"Loading VAE...")
    vae = AutoencoderKL.from_pretrained(VAE_MODEL).to(device)
    vae.eval()

    # Load episodes
    ep_files = sorted(Path(args.data).glob("episode_*.pt"))[:args.num_episodes]
    print(f"Loading {len(ep_files)} episodes...")
    episodes = [torch.load(f, weights_only=True) for f in ep_files]

    # W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="quake3-worldmodel", entity="rzhang139",
            name="eval-latent-medium-60ep",
            config={"model_size": args.model_size, "cfg_scale": args.cfg_scale,
                    "rollout_len": args.rollout_len, "num_episodes": args.num_episodes},
        )

    # Teacher-forced PSNR
    print("Running teacher-forced eval...")
    tf_psnr, tf_copy = eval_teacher_forced(model, vae, episodes, device, cfg_scale=args.cfg_scale)
    print(f"Teacher-forced PSNR: {tf_psnr:.2f} dB  (copy baseline: {tf_copy:.2f} dB)")

    # Autoregressive PSNR curve
    print("Running autoregressive eval...")
    ar_psnr, ar_copy = eval_autoregressive(
        model, vae, episodes, device, rollout_len=args.rollout_len, cfg_scale=args.cfg_scale
    )
    print("Autoregressive PSNR curve:")
    for step, p in ar_psnr.items():
        print(f"  frame {step:2d}: pred={p:.2f} dB  copy={ar_copy.get(step, 0):.2f} dB")

    # Generate comparison videos
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    video_paths = []
    for i in range(min(args.num_videos, len(episodes))):
        print(f"Generating video {i+1}/{args.num_videos}...")
        frames = make_comparison_video(
            model, vae, episodes[i], device,
            rollout_len=args.rollout_len, cfg_scale=args.cfg_scale
        )
        path = out_dir / f"rollout_{i:02d}.mp4"
        import imageio
        imageio.mimwrite(str(path), frames, fps=10, quality=8)
        video_paths.append(path)
        print(f"  Saved: {path} ({frames.shape[0]} frames, {frames.shape[2]}x{frames.shape[1]})")

    # Log to W&B
    if wandb_run:
        import wandb
        log = {
            "tf_psnr": tf_psnr,
            "tf_copy_psnr": tf_copy,
            "tf_psnr_above_copy": tf_psnr - tf_copy,
        }
        for step, p in ar_psnr.items():
            log[f"ar_psnr_frame{step:02d}"] = p
        for step, p in ar_copy.items():
            log[f"ar_copy_frame{step:02d}"] = p
        for p in video_paths:
            log[f"video_{p.stem}"] = wandb.Video(str(p), fps=10, format="mp4")
        wandb_run.log(log)
        wandb_run.finish()
        print("Results logged to W&B.")

    print("\n=== Summary ===")
    print(f"Teacher-forced PSNR:  {tf_psnr:.2f} dB  (copy: {tf_copy:.2f} dB, delta: {tf_psnr-tf_copy:+.2f})")
    print(f"AR PSNR @ frame 1:    {ar_psnr.get(1, 0):.2f} dB")
    print(f"AR PSNR @ frame 8:    {ar_psnr.get(8, 0):.2f} dB")
    print(f"AR PSNR @ frame 32:   {ar_psnr.get(32, 0):.2f} dB")
    print(f"Videos: {[str(p) for p in video_paths]}")


if __name__ == "__main__":
    main()
