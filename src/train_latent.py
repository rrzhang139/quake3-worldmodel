"""Train world model in VAE latent space.

Same EDM diffusion U-Net, but operates on (4, 15, 20) latent frames
instead of (3, 120, 160) pixel frames. 48x fewer values per frame.

Data flow:
  Pre-encoded latents (4, 15, 20) float32
    → dataset returns context (L, 4, 15, 20) + target (4, 15, 20) + action
    → model.training_loss(): add noise to target latent, predict clean latent
    → same EDM preconditioning, same sigma schedule, same loss
    → at inference: model.sample() generates latent → VAE decode → pixels

Usage:
    python src/train_latent.py --data data/latents_160x120_5k --epochs 30 --batch_size 128 --wandb
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from model import make_denoiser
from dataset_latent import make_latent_dataloader


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # bf16 mixed precision: ~1.5-2x faster on Ampere (A40, A100, RTX 3090+)
    # bf16 has same exponent range as fp32 — no gradient underflow, no scaler needed
    use_amp = device.type == "cuda" and args.bf16
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"Mixed precision: {'bf16' if use_amp else 'fp32'}")

    # Data — latent frames are (4, 15, 20) float32
    dataloader = make_latent_dataloader(
        args.data,
        batch_size=args.batch_size,
        num_context_frames=args.num_context,
        num_workers=args.num_workers,
    )
    print(f"Dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches/epoch")

    # Model — same architecture, just img_channels=4 for latent space
    # Latent spatial dims are 15x20, but img_size is used for some internal sizing
    model = make_denoiser(
        num_actions=args.num_actions,
        img_size=20,  # width of latent (used for padding calc)
        img_channels=4,  # 4 latent channels (not 3 RGB)
        num_context_frames=args.num_context,
        model_size=args.model_size,
        cfg_drop_prob=args.cfg_drop_prob,
        action_aux_weight=args.action_aux_weight,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_size} ({params:,} params), latent mode (4ch, 15x20)")

    # torch.compile: ~20-30% additional speedup via kernel fusion (PyTorch 2.0+)
    # Known issue: bf16 + compile can cause dtype mismatch in some PyTorch versions.
    # If training crashes on first step, rerun with --compile removed.
    if args.compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile (first step will be slow ~60s)...")
        model = torch.compile(model, mode="reduce-overhead")

    # Optimizer (DIAMOND recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # LR warmup
    if args.lr_warmup_steps > 0:
        def lr_lambda(step):
            return min(1.0, step / args.lr_warmup_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    else:
        scheduler = None

    # W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="quake3-worldmodel",
            entity="rzhang139",
            config=vars(args),
            name=f"latent_{args.model_size}_{args.batch_size}bs",
        )

    # Resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["step"]
        print(f"Resumed from {args.resume} (epoch {start_epoch}, step {global_step})")

    # Output
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []

        for batch in dataloader:
            context = batch["context"].to(device)  # (B, L, 4, 15, 20)
            target = batch["target"].to(device)     # (B, 4, 15, 20)
            action = batch["action"].to(device)     # (B,)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                loss = model.training_loss(
                    context, target, action,
                    noise_aug_sigma=args.noise_aug,
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_losses.append(loss.item())
            global_step += 1

            if global_step % args.log_every == 0:
                avg_loss = np.mean(epoch_losses[-args.log_every:])
                elapsed = time.time() - t0
                print(f"  step {global_step}, loss={avg_loss:.4f}, elapsed={elapsed:.0f}s")
                if wandb_run:
                    wandb_run.log({"loss": avg_loss, "step": global_step, "epoch": epoch})

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, elapsed={elapsed:.0f}s")

        if wandb_run:
            wandb_run.log({"epoch_loss": avg_loss, "epoch": epoch})

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "loss": float(avg_loss),
            }, out_dir / "best.pt")

        if (epoch + 1) % args.save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "step": global_step,
                "loss": float(avg_loss),
            }, out_dir / f"checkpoint_e{epoch+1}.pt")

    if wandb_run:
        wandb_run.finish()

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Checkpoint: {out_dir / 'best.pt'}")


def main():
    parser = argparse.ArgumentParser(description="Train latent-space world model")
    # Data
    parser.add_argument("--data", type=str, required=True, help="Dir with pre-encoded latent episodes")
    parser.add_argument("--num_context", type=int, default=4)
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=2)
    # Model
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large"])
    parser.add_argument("--cfg_drop_prob", type=float, default=0.15)
    parser.add_argument("--action_aux_weight", type=float, default=0.1)
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr_warmup_steps", type=int, default=100)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--noise_aug", type=float, default=0.3)
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--output", type=str, default="experiments/run_latent")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--bf16", action="store_true", help="Use bf16 mixed precision (Ampere+, ~1.5-2x speedup)")
    parser.add_argument("--compile", action="store_true", help="torch.compile for extra ~20-30%% speedup")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
