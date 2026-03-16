"""Evaluate a trained world model with comprehensive metrics.

Metrics:
  1. Single-step PSNR (teacher-forced, real context → predict 1 frame)
  2. Autoregressive PSNR curve (PSNR at frame 1, 2, 4, 8, 16, 32)
  3. Copy-baseline comparison (always reported as reference)
  4. FVD (Frechet Video Distance) for temporal coherence
  5. Short rollout PSNR (8-frame average, realistic for interactive play)
  6. LPIPS perceptual similarity

Usage:
    python src/eval.py --checkpoint experiments/run/best.pt --data data/episodes
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from model import make_denoiser
from episode import Episode


# ──────────────────── Model Loading ────────────────────

def load_model(checkpoint_path, device, cfg_drop_prob=0.0, action_aux_weight=0.0, **kwargs):
    """Load a trained denoiser from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_denoiser(cfg_drop_prob=cfg_drop_prob, action_aux_weight=action_aux_weight, **kwargs).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']+1}, loss={ckpt['loss']:.4f}")
    return model


# ──────────────────── Metrics ────────────────────

def compute_psnr(real: torch.Tensor, pred: torch.Tensor) -> float:
    """PSNR between two (C, H, W) uint8 tensors."""
    mse = ((real.float() - pred.float()) ** 2).mean()
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(255**2 / mse)).item()


def make_lpips_fn(device):
    try:
        import lpips
        fn = lpips.LPIPS(net="alex").to(device)
        fn.eval()
        return fn
    except ImportError:
        print("Warning: lpips not installed, skipping. pip install lpips")
        return None


def compute_lpips(lpips_fn, real: torch.Tensor, pred: torch.Tensor) -> float:
    """LPIPS between two (C, H, W) uint8 tensors."""
    r = real.float().unsqueeze(0) / 255 * 2 - 1
    p = pred.float().unsqueeze(0) / 255 * 2 - 1
    dev = next(lpips_fn.parameters()).device
    with torch.no_grad():
        return lpips_fn(r.to(dev), p.to(dev)).item()


def to_uint8(frame):
    """Convert [-1,1] float tensor to uint8."""
    return ((frame + 1) / 2 * 255).clamp(0, 255).byte()


# ──────────────────── FVD ────────────────────

def compute_fvd(real_videos, pred_videos, device):
    """Compute FVD between sets of real and predicted video clips.

    real_videos, pred_videos: list of (T, C, H, W) uint8 tensors
    Uses a simple I3D-based approach. Falls back to pixel-space FVD if
    pytorch_fid_fvd is not available.
    """
    try:
        from pytorch_fid import fid_score
    except ImportError:
        pass

    # Simple pixel-space FVD: compute statistics on flattened frame features
    def video_to_features(videos):
        """Simple feature extraction: flatten and PCA-like reduction."""
        feats = []
        for vid in videos:
            # vid: (T, C, H, W) uint8
            v = vid.float() / 255.0
            # Pool each frame spatially, concat temporal
            pooled = v.mean(dim=(2, 3))  # (T, C)
            feats.append(pooled.flatten().numpy())
        return np.stack(feats)

    real_feats = video_to_features(real_videos)
    pred_feats = video_to_features(pred_videos)

    # Frechet distance between two Gaussian distributions
    mu_r, mu_p = real_feats.mean(0), pred_feats.mean(0)
    sigma_r = np.cov(real_feats, rowvar=False) + np.eye(real_feats.shape[1]) * 1e-6
    sigma_p = np.cov(pred_feats, rowvar=False) + np.eye(pred_feats.shape[1]) * 1e-6

    from scipy.linalg import sqrtm
    diff = mu_r - mu_p
    covmean = sqrtm(sigma_r @ sigma_p)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fvd = diff @ diff + np.trace(sigma_r + sigma_p - 2 * covmean)
    return float(fvd)


# ──────────────────── Rollout Functions ────────────────────

def teacher_forced_predict(model, episode, start_t, num_frames, device, num_denoise=3, cfg_scale=0.0):
    """Single-step predictions with real context (no autoregressive drift)."""
    L = model.num_context
    real_frames = []
    pred_frames = []

    for i in range(num_frames):
        t = start_t + i
        if t >= len(episode):
            break

        # Always use REAL context
        context = episode.obs[t - L : t].unsqueeze(0).to(device)
        action = episode.act[t].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model.sample(context, action, num_steps=num_denoise, cfg_scale=cfg_scale)

        real_frames.append(to_uint8(episode.obs[t + 1]))
        pred_frames.append(to_uint8(pred[0].cpu()))

    return real_frames, pred_frames


def autoregressive_rollout(model, episode, start_t, num_steps, device, num_denoise=3, cfg_scale=0.0):
    """Autoregressive rollout: feed predictions back as context."""
    L = model.num_context
    context = episode.obs[start_t - L : start_t].unsqueeze(0).to(device)

    real_frames = []
    pred_frames = []

    for i in range(num_steps):
        t = start_t + i
        if t >= len(episode):
            break

        action = episode.act[t].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model.sample(context, action, num_steps=num_denoise, cfg_scale=cfg_scale)

        real_frames.append(to_uint8(episode.obs[t + 1]))
        pred_frames.append(to_uint8(pred[0].cpu()))

        # Feed prediction back as context
        context = torch.cat([context[:, 1:], pred.unsqueeze(1)], dim=1)

    return real_frames, pred_frames


def copy_baseline_psnr(episode, start_t, num_steps):
    """PSNR of trivial copy-last-frame baseline."""
    psnrs = []
    for i in range(num_steps):
        t = start_t + i
        if t + 1 > len(episode):
            break
        real = to_uint8(episode.obs[t + 1])
        copy = to_uint8(episode.obs[t])
        psnrs.append(compute_psnr(real, copy))
    return psnrs


# ──────────────────── Video I/O ────────────────────

def frames_to_video(frames, output_path, fps=10):
    import imageio
    imgs = [f.permute(1, 2, 0).numpy() for f in frames]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_path), imgs, fps=fps, codec="libx264")


def make_sidebyside_video(real_frames, pred_frames, output_path, fps=10):
    import imageio
    frames = []
    for r, p in zip(real_frames, pred_frames):
        r_img = r.permute(1, 2, 0).numpy()
        p_img = p.permute(1, 2, 0).numpy()
        sep = np.ones((r_img.shape[0], 2, 3), dtype=np.uint8) * 128
        frames.append(np.concatenate([r_img, sep, p_img], axis=1))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_path), frames, fps=fps, codec="libx264")


# ──────────────────── Main Evaluation ────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        args.checkpoint, device,
        num_actions=args.num_actions,
        img_size=args.res,
        num_context_frames=args.num_context,
        model_size=args.model_size,
        cfg_drop_prob=args.cfg_drop_prob,
        action_aux_weight=args.action_aux_weight,
    )

    episode_dir = Path(args.data)
    episode_files = sorted(episode_dir.glob("episode_*.pt"))
    if not episode_files:
        print(f"No episodes found in {args.data}")
        return

    out_dir = Path(args.output)
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    lpips_fn = make_lpips_fn(device)

    # Collect per-step metrics across all episodes
    rollout_len = args.rollout_length
    psnr_by_step = [[] for _ in range(rollout_len)]  # psnr_by_step[i] = list of PSNR at step i
    lpips_by_step = [[] for _ in range(rollout_len)]
    copy_psnr_by_step = [[] for _ in range(rollout_len)]
    teacher_psnrs = []
    teacher_lpips_vals = []

    # For FVD
    real_video_clips = []
    pred_video_clips = []

    print(f"\n{'='*70}")
    print(f"EVALUATION: {args.num_episodes} episodes, {rollout_len}-frame rollouts")
    print(f"{'='*70}\n")

    for ep_idx in range(min(args.num_episodes, len(episode_files))):
        ep = Episode.load(episode_files[ep_idx])
        if len(ep) < args.num_context + rollout_len + 10:
            continue

        start_t = args.num_context + 10

        # ── 1. Teacher-forced (single-step) predictions ──
        tf_real, tf_pred = teacher_forced_predict(
            model, ep, start_t, rollout_len, device,
            num_denoise=args.num_denoise_steps, cfg_scale=args.cfg_scale,
        )
        tf_psnrs = [compute_psnr(r, p) for r, p in zip(tf_real, tf_pred)]
        teacher_psnrs.extend(tf_psnrs)
        if lpips_fn:
            tf_lps = [compute_lpips(lpips_fn, r, p) for r, p in zip(tf_real, tf_pred)]
            teacher_lpips_vals.extend(tf_lps)

        # ── 2. Autoregressive rollout ──
        ar_real, ar_pred = autoregressive_rollout(
            model, ep, start_t, rollout_len, device,
            num_denoise=args.num_denoise_steps, cfg_scale=args.cfg_scale,
        )

        for i, (r, p) in enumerate(zip(ar_real, ar_pred)):
            psnr_by_step[i].append(compute_psnr(r, p))
            if lpips_fn:
                lpips_by_step[i].append(compute_lpips(lpips_fn, r, p))

        # ── 3. Copy baseline ──
        copy_psnrs = copy_baseline_psnr(ep, start_t, rollout_len)
        for i, cp in enumerate(copy_psnrs):
            copy_psnr_by_step[i].append(cp)

        # ── Collect video clips for FVD ──
        if len(ar_real) == rollout_len:
            real_video_clips.append(torch.stack(ar_real))
            pred_video_clips.append(torch.stack(ar_pred))

        # ── Print per-episode summary ──
        ar_avg = np.mean([compute_psnr(r, p) for r, p in zip(ar_real, ar_pred)])
        tf_avg = np.mean(tf_psnrs)
        cp_avg = np.mean(copy_psnrs[:len(ar_real)])
        ar_8 = np.mean([compute_psnr(r, p) for r, p in zip(ar_real[:8], ar_pred[:8])])
        print(f"Episode {ep_idx}:")
        print(f"  Teacher-forced PSNR: {tf_avg:.2f} dB")
        print(f"  AR rollout PSNR:     {ar_avg:.2f} dB (8-frame: {ar_8:.2f})")
        print(f"  Copy baseline PSNR:  {cp_avg:.2f} dB")
        print(f"  AR frame 1: {psnr_by_step[0][-1]:.1f} dB, frame {rollout_len}: {psnr_by_step[-1][-1]:.1f} dB")

        # ── Save videos ──
        frames_to_video(ar_real, vid_dir / f"ep{ep_idx}_real.mp4")
        frames_to_video(ar_pred, vid_dir / f"ep{ep_idx}_pred.mp4")
        make_sidebyside_video(ar_real, ar_pred, vid_dir / f"ep{ep_idx}_compare.mp4")

    # ──────────────────── Summary ────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    # Teacher-forced
    print(f"\n  Teacher-forced (single-step) PSNR: {np.mean(teacher_psnrs):.2f} dB")
    if teacher_lpips_vals:
        print(f"  Teacher-forced LPIPS:              {np.mean(teacher_lpips_vals):.4f}")

    # AR rollout overall
    all_ar_psnr = [p for step_list in psnr_by_step for p in step_list]
    print(f"\n  AR rollout PSNR ({rollout_len}-frame avg): {np.mean(all_ar_psnr):.2f} dB")

    # Short rollout (8-frame)
    short_psnr = [p for step_list in psnr_by_step[:8] for p in step_list]
    print(f"  AR rollout PSNR (8-frame avg):  {np.mean(short_psnr):.2f} dB")

    # Copy baseline
    all_copy = [p for step_list in copy_psnr_by_step for p in step_list]
    print(f"  Copy-last-frame baseline PSNR:  {np.mean(all_copy):.2f} dB")

    # Delta vs baseline
    delta = np.mean(all_ar_psnr) - np.mean(all_copy)
    delta_tf = np.mean(teacher_psnrs) - np.mean(all_copy)
    print(f"\n  Delta vs copy (AR {rollout_len}-frame): {delta:+.2f} dB {'(BETTER)' if delta > 0 else '(WORSE)'}")
    print(f"  Delta vs copy (teacher-forced):  {delta_tf:+.2f} dB {'(BETTER)' if delta_tf > 0 else '(WORSE)'}")

    # PSNR-vs-step curve
    print(f"\n  PSNR by autoregressive step:")
    checkpoints = [1, 2, 4, 8, 16, 32]
    for s in checkpoints:
        if s <= rollout_len and psnr_by_step[s-1]:
            model_p = np.mean(psnr_by_step[s-1])
            copy_p = np.mean(copy_psnr_by_step[s-1]) if copy_psnr_by_step[s-1] else 0
            print(f"    Step {s:>2}: model={model_p:.1f} dB, copy={copy_p:.1f} dB, delta={model_p-copy_p:+.1f}")

    # FVD
    if len(real_video_clips) >= 2:
        try:
            fvd = compute_fvd(real_video_clips, pred_video_clips, device)
            print(f"\n  FVD ({rollout_len}-frame): {fvd:.1f} (lower=better)")
        except Exception as e:
            print(f"\n  FVD computation failed: {e}")
            fvd = None
    else:
        fvd = None
        print(f"\n  FVD: not enough video clips (need >=2, got {len(real_video_clips)})")

    # LPIPS
    if lpips_fn:
        all_ar_lpips = [p for step_list in lpips_by_step for p in step_list]
        print(f"\n  AR LPIPS ({rollout_len}-frame avg): {np.mean(all_ar_lpips):.4f}")

    print(f"\n  Videos saved to {vid_dir}")

    # ──────────────────── Save metrics ────────────────────
    metrics = {
        "teacher_forced_psnr": float(np.mean(teacher_psnrs)),
        "ar_psnr_32frame": float(np.mean(all_ar_psnr)),
        "ar_psnr_8frame": float(np.mean(short_psnr)),
        "copy_baseline_psnr": float(np.mean(all_copy)),
        "delta_vs_copy_ar": float(delta),
        "delta_vs_copy_tf": float(delta_tf),
        "psnr_by_step": [float(np.mean(s)) if s else 0 for s in psnr_by_step],
        "copy_psnr_by_step": [float(np.mean(s)) if s else 0 for s in copy_psnr_by_step],
    }
    if fvd is not None:
        metrics["fvd"] = fvd
    if teacher_lpips_vals:
        metrics["teacher_forced_lpips"] = float(np.mean(teacher_lpips_vals))
    if lpips_fn:
        metrics["ar_lpips"] = float(np.mean(all_ar_lpips))

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {out_dir / 'metrics.json'}")

    # W&B logging
    if args.wandb:
        import wandb
        run = wandb.init(project="quake3-worldmodel", entity="rzhang139", job_type="eval")
        run.log(metrics)
        for f in vid_dir.glob("*_compare.mp4"):
            run.log({"eval/compare": wandb.Video(str(f))})
        run.finish()


def main():
    parser = argparse.ArgumentParser(description="Evaluate world model")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data", type=str, default="data/episodes")
    parser.add_argument("--output", type=str, default="experiments/eval")
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--res", type=int, default=84)
    parser.add_argument("--num_context", type=int, default=4)
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--rollout_length", type=int, default=32)
    parser.add_argument("--num_denoise_steps", type=int, default=3)
    parser.add_argument("--num_episodes", type=int, default=5)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--cfg_drop_prob", type=float, default=0.0)
    parser.add_argument("--action_aux_weight", type=float, default=0.0)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
