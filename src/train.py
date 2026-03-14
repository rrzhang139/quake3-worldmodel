"""Training loop for the world model denoiser.

Usage:
    python src/train.py --data data/episodes --epochs 100
    python src/train.py --data data/episodes --epochs 500 --model_size medium --wandb
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

from model import make_denoiser
from dataset import make_dataloader


def save_checkpoint(model, optimizer, epoch, step, loss, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "loss": float(loss),
    }, path)


def generate_samples(model, dataloader, device, num_samples=4, num_steps=3):
    """Generate sample predictions for visualization."""
    model.eval()
    batch = next(iter(dataloader))
    context = batch["context"][:num_samples].to(device)
    target = batch["target"][:num_samples].to(device)
    action = batch["action"][:num_samples].to(device)

    with torch.no_grad():
        predicted = model.sample(context, action, num_steps=num_steps)

    model.train()
    # Return as uint8 images for logging
    target_img = ((target + 1) / 2 * 255).clamp(0, 255).byte().cpu()
    pred_img = ((predicted + 1) / 2 * 255).clamp(0, 255).byte().cpu()
    context_img = ((context[:, -1] + 1) / 2 * 255).clamp(0, 255).byte().cpu()
    return context_img, target_img, pred_img


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    dataloader = make_dataloader(
        args.data,
        batch_size=args.batch_size,
        num_context_frames=args.num_context,
        num_workers=args.num_workers,
        streaming=args.streaming,
    )
    print(f"Dataset: {len(dataloader.dataset)} samples, {len(dataloader)} batches/epoch")

    # Model
    model = make_denoiser(
        num_actions=args.num_actions,
        img_size=args.res,
        num_context_frames=args.num_context,
        model_size=args.model_size,
        cfg_drop_prob=args.cfg_drop_prob,
        action_aux_weight=args.action_aux_weight,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model_size} ({params:,} params)")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # W&B
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project="quake3-worldmodel",
            entity="rzhang139",
            config=vars(args),
            name=f"{args.model_size}_{args.res}x{args.res}",
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

    # Output dir
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    model.train()
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_losses = []

        for batch in dataloader:
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            action = batch["action"].to(device)

            loss = model.training_loss(
                context, target, action,
                noise_aug_sigma=args.noise_aug,
            )

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_losses.append(loss.item())
            global_step += 1

            # Log every N steps
            if global_step % args.log_every == 0:
                avg_loss = np.mean(epoch_losses[-args.log_every:])
                elapsed = time.time() - t0
                print(f"  step {global_step}, loss={avg_loss:.4f}, "
                      f"elapsed={elapsed:.0f}s")
                if wandb_run:
                    wandb_run.log({"loss": avg_loss, "step": global_step, "epoch": epoch})

        # Epoch summary
        avg_loss = np.mean(epoch_losses)
        elapsed = time.time() - t0
        print(f"Epoch {epoch+1}/{args.epochs}: loss={avg_loss:.4f}, "
              f"elapsed={elapsed:.0f}s")

        if wandb_run:
            wandb_run.log({"epoch_loss": avg_loss, "epoch": epoch})

        # Generate samples periodically
        if (epoch + 1) % args.sample_every == 0:
            ctx_img, tgt_img, pred_img = generate_samples(
                model, dataloader, device, num_samples=4
            )
            if wandb_run:
                import wandb
                images = []
                for i in range(min(4, ctx_img.shape[0])):
                    # Create side-by-side: context | target | predicted
                    row = torch.cat([ctx_img[i], tgt_img[i], pred_img[i]], dim=2)
                    images.append(wandb.Image(
                        row.permute(1, 2, 0).numpy(),
                        caption=f"ctx | target | pred (action={batch['action'][i].item()})"
                    ))
                wandb_run.log({"samples": images, "epoch": epoch})
            # Save sample images locally too
            from PIL import Image
            for i in range(min(2, ctx_img.shape[0])):
                row = torch.cat([ctx_img[i], tgt_img[i], pred_img[i]], dim=2)
                img = Image.fromarray(row.permute(1, 2, 0).numpy())
                img.save(out_dir / f"sample_e{epoch+1}_{i}.png")

        # Checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, global_step, avg_loss,
                out_dir / f"checkpoint_e{epoch+1}.pt",
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, epoch, global_step, avg_loss,
                out_dir / "best.pt",
            )

    # Final save
    save_checkpoint(
        model, optimizer, args.epochs - 1, global_step, avg_loss,
        out_dir / "final.pt",
    )
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")

    if wandb_run:
        wandb_run.finish()


def main():
    parser = argparse.ArgumentParser(description="Train world model denoiser")
    # Data
    parser.add_argument("--data", type=str, default="data/episodes")
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--res", type=int, default=84)
    parser.add_argument("--num_context", type=int, default=4)
    # Model
    parser.add_argument("--model_size", type=str, default="small",
                        choices=["tiny", "small", "medium", "large"])
    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise_aug", type=float, default=0.0,
                        help="GameNGen context noise augmentation sigma")
    parser.add_argument("--cfg_drop_prob", type=float, default=0.0,
                        help="Classifier-free guidance: probability of dropping action (0.1 = 10%)")
    parser.add_argument("--action_aux_weight", type=float, default=0.0,
                        help="Weight for action prediction auxiliary loss")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--streaming", action="store_true",
                        help="Stream episodes from disk instead of loading all into RAM")
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--sample_every", type=int, default=10)
    parser.add_argument("--save_every", type=int, default=25)
    parser.add_argument("--output", type=str, default="experiments/run")
    # Resume
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
