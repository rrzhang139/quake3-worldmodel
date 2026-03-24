"""Pre-encode episode frames through SD VAE into latent space.

Takes episode .pt files (uint8 obs) and produces latent .pt files
with the same structure but obs replaced by VAE latent codes.

Data flow per episode:
  obs (T+1, 3, 120, 160) uint8 → float [-1,1]
    → SD VAE encoder → (T+1, 4, 15, 20) float latents
    → scale by 0.18215 (SD convention)
    → save as new .pt with {obs_latent, act, rew, end, trunc}

Usage:
    python src/encode_latents.py --input data/episodes_160x120_5k --output data/latents_160x120_5k
"""

import argparse
import time
from pathlib import Path

import torch
from diffusers import AutoencoderKL


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Frames to encode at once (higher = faster, more RAM)")
    parser.add_argument("--vae_model", type=str, default="stabilityai/sd-vae-ft-mse")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load VAE
    print(f"Loading VAE from {args.vae_model}...")
    vae = AutoencoderKL.from_pretrained(args.vae_model).to(device)
    vae.eval()
    # SD latent scaling factor
    scale_factor = 0.18215

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_files = sorted(input_dir.glob("episode_*.pt"))
    print(f"Found {len(episode_files)} episodes in {input_dir}")

    t0 = time.time()
    total_frames = 0

    for i, f in enumerate(episode_files):
        d = torch.load(f, weights_only=True)
        obs_uint8 = d["obs"]  # (T+1, C, H, W) uint8

        # Convert to float [-1, 1]
        obs_float = obs_uint8.float().div_(255).mul_(2).sub_(1)
        T_plus_1 = obs_float.shape[0]

        # Encode in batches
        all_latents = []
        for start in range(0, T_plus_1, args.batch_size):
            batch = obs_float[start:start + args.batch_size].to(device)
            with torch.no_grad():
                posterior = vae.encode(batch)
                latent = posterior.latent_dist.mode()  # deterministic
                latent = latent * scale_factor
            all_latents.append(latent.cpu())

        latents = torch.cat(all_latents, dim=0)  # (T+1, 4, H/8, W/8)

        # Save with same structure, replacing obs with latents
        out = {
            "obs_latent": latents,  # (T+1, 4, 15, 20) float32
            "act": d["act"],
            "rew": d["rew"],
            "end": d["end"],
            "trunc": d["trunc"],
        }
        torch.save(out, output_dir / f.name)

        total_frames += T_plus_1
        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - t0
            fps = total_frames / elapsed
            print(f"  {i+1}/{len(episode_files)}: {total_frames} frames, "
                  f"{fps:.0f} fps, {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone! {len(episode_files)} episodes, {total_frames} frames in {elapsed:.1f}s")
    print(f"Latent shape per frame: {latents.shape[1:]}")
    print(f"Saved to {output_dir}")

    # Print size comparison
    import os
    input_size = sum(os.path.getsize(f) for f in episode_files) / 1e9
    output_size = sum(os.path.getsize(output_dir / f.name) for f in episode_files) / 1e9
    print(f"Input size: {input_size:.1f} GB → Output size: {output_size:.1f} GB "
          f"({output_size/input_size:.1%})")


if __name__ == "__main__":
    main()
