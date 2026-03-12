"""Evaluate a trained world model by generating rollout videos.

Usage:
    python src/eval.py --checkpoint experiments/run1/best.pt --data data/episodes --output experiments/run1/eval
"""

import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from model import make_denoiser
from episode import Episode


def load_model(checkpoint_path, device, **kwargs):
    """Load a trained denoiser from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = make_denoiser(**kwargs).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt['epoch']+1}, loss={ckpt['loss']:.4f}")
    return model


def autoregressive_rollout(model, seed_episode, start_t, num_steps, device, num_denoise=3):
    """Generate frames autoregressively from a seed episode."""
    L = model.num_context

    context = seed_episode.obs[start_t - L : start_t].unsqueeze(0).to(device)

    real_frames = []
    pred_frames = []

    for i in range(num_steps):
        t = start_t + i
        if t >= len(seed_episode):
            break

        action = seed_episode.act[t].unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model.sample(context, action, num_steps=num_denoise)

        real_frame = seed_episode.obs[t + 1]
        real_frames.append(((real_frame + 1) / 2 * 255).clamp(0, 255).byte())
        pred_frames.append(((pred[0].cpu() + 1) / 2 * 255).clamp(0, 255).byte())

        context = torch.cat([context[:, 1:], pred.unsqueeze(1)], dim=1)

    return real_frames, pred_frames


def compute_psnr(real: torch.Tensor, pred: torch.Tensor) -> float:
    mse = ((real.float() - pred.float()) ** 2).mean()
    if mse == 0:
        return float("inf")
    return (10 * torch.log10(255**2 / mse)).item()


def make_lpips_fn(device):
    """Create LPIPS perceptual similarity function. Returns None if lpips not installed."""
    try:
        import lpips
        fn = lpips.LPIPS(net="alex").to(device)
        fn.eval()
        return fn
    except ImportError:
        print("Warning: lpips not installed, skipping perceptual metrics. pip install lpips")
        return None


def compute_lpips(lpips_fn, real: torch.Tensor, pred: torch.Tensor) -> float:
    """Compute LPIPS between two (C, H, W) uint8 tensors. Returns scalar."""
    # LPIPS expects (B, C, H, W) float in [-1, 1]
    r = real.float().unsqueeze(0) / 255 * 2 - 1
    p = pred.float().unsqueeze(0) / 255 * 2 - 1
    with torch.no_grad():
        return lpips_fn(r.to(lpips_fn.parameters().__next__().device),
                        p.to(lpips_fn.parameters().__next__().device)).item()


def frames_to_video(frames, output_path, fps=10):
    """Write a list of (C, H, W) uint8 tensors to an MP4."""
    import imageio
    imgs = [f.permute(1, 2, 0).numpy() for f in frames]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_path), imgs, fps=fps, codec="libx264")


def make_sidebyside_video(real_frames, pred_frames, output_path, fps=10):
    """Side-by-side comparison video: real | pred."""
    import imageio
    frames = []
    for r, p in zip(real_frames, pred_frames):
        r_img = r.permute(1, 2, 0).numpy()
        p_img = p.permute(1, 2, 0).numpy()
        sep = np.ones((r_img.shape[0], 2, 3), dtype=np.uint8) * 128
        frames.append(np.concatenate([r_img, sep, p_img], axis=1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(output_path), frames, fps=fps, codec="libx264")


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(
        args.checkpoint, device,
        num_actions=args.num_actions,
        img_size=args.res,
        num_context_frames=args.num_context,
        model_size=args.model_size,
    )

    episode_dir = Path(args.data)
    episode_files = sorted(episode_dir.glob("episode_*.pt"))
    if not episode_files:
        print(f"No episodes found in {args.data}")
        return

    out_dir = Path(args.output)
    vid_dir = out_dir / "videos"
    vid_dir.mkdir(parents=True, exist_ok=True)

    all_psnr = []
    all_lpips = []

    # Try to load LPIPS
    lpips_fn = make_lpips_fn(device)

    for ep_idx in range(min(args.num_episodes, len(episode_files))):
        ep = Episode.load(episode_files[ep_idx])
        if len(ep) < args.num_context + args.rollout_length:
            continue

        start_t = args.num_context + 10
        real_frames, pred_frames = autoregressive_rollout(
            model, ep, start_t, args.rollout_length, device,
            num_denoise=args.num_denoise_steps,
        )

        # PSNR
        psnrs = [compute_psnr(r, p) for r, p in zip(real_frames, pred_frames)]
        avg_psnr = np.mean(psnrs)
        all_psnr.extend(psnrs)

        # LPIPS
        if lpips_fn is not None:
            lps = [compute_lpips(lpips_fn, r, p) for r, p in zip(real_frames, pred_frames)]
            avg_lp = np.mean(lps)
            all_lpips.extend(lps)
            print(f"Episode {ep_idx}: PSNR={avg_psnr:.2f} dB, LPIPS={avg_lp:.4f} "
                  f"(frame 1: {psnrs[0]:.1f}/{lps[0]:.3f}, frame {len(psnrs)}: {psnrs[-1]:.1f}/{lps[-1]:.3f})")
        else:
            print(f"Episode {ep_idx}: avg PSNR={avg_psnr:.2f} dB "
                  f"(frame 1: {psnrs[0]:.1f}, frame {len(psnrs)}: {psnrs[-1]:.1f})")

        # Save 3 videos: real, predicted, side-by-side
        frames_to_video(real_frames, vid_dir / f"ep{ep_idx}_real.mp4")
        frames_to_video(pred_frames, vid_dir / f"ep{ep_idx}_pred.mp4")
        make_sidebyside_video(real_frames, pred_frames, vid_dir / f"ep{ep_idx}_compare.mp4")
        print(f"  Saved: ep{ep_idx}_real.mp4, ep{ep_idx}_pred.mp4, ep{ep_idx}_compare.mp4")

    print(f"\nOverall: avg PSNR={np.mean(all_psnr):.2f} dB over {len(all_psnr)} frames")
    if all_lpips:
        print(f"Overall: avg LPIPS={np.mean(all_lpips):.4f} (lower=better)")
    print(f"Videos saved to {vid_dir}")

    # Save metrics to file
    metrics = {"psnr_mean": float(np.mean(all_psnr)), "psnr_per_frame": [float(x) for x in all_psnr]}
    if all_lpips:
        metrics["lpips_mean"] = float(np.mean(all_lpips))
        metrics["lpips_per_frame"] = [float(x) for x in all_lpips]
    import json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if args.wandb:
        import wandb
        run = wandb.init(project="quake3-worldmodel", entity="rzhang139", job_type="eval")
        log_dict = {"eval/psnr_mean": np.mean(all_psnr)}
        if all_lpips:
            log_dict["eval/lpips_mean"] = np.mean(all_lpips)
        run.log(log_dict)
        for f in vid_dir.glob("*_compare.mp4"):
            run.log({"eval/compare": wandb.Video(str(f))})
        for f in vid_dir.glob("*_pred.mp4"):
            run.log({"eval/predicted": wandb.Video(str(f))})
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
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
