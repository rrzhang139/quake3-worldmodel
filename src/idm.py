"""Inverse Dynamics Model (IDM): predicts action from (frame_t, frame_t+1).

Used as an eval metric: if a trained IDM achieves high accuracy on model-generated
transitions, it means the world model produces action-consistent frame dynamics.

Usage:
    # Train IDM on real episodes
    python src/idm.py train --data data/episodes --output experiments/idm

    # Evaluate IDM on world model rollouts
    python src/idm.py eval --idm_checkpoint experiments/idm/best.pt \
        --wm_checkpoint experiments/run/best.pt --data data/episodes
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader

from episode import Episode


class IDMDataset(Dataset):
    """Samples (frame_t, frame_t+1, action) from episodes."""

    def __init__(self, episode_dir: str, split: str = "train", train_frac: float = 0.9):
        episode_files = sorted(Path(episode_dir).glob("episode_*.pt"))
        if not episode_files:
            raise FileNotFoundError(f"No episodes in {episode_dir}")

        n_train = int(len(episode_files) * train_frac)
        if split == "train":
            episode_files = episode_files[:n_train]
        else:
            episode_files = episode_files[n_train:]

        self.samples = []  # list of (frame_t, frame_t+1, action)
        for f in episode_files:
            ep = Episode.load(f)
            for t in range(len(ep)):
                if ep.end[t]:
                    continue  # skip terminal transitions
                self.samples.append((ep.obs[t], ep.obs[t + 1], ep.act[t]))

        print(f"IDM {split}: {len(self.samples)} transitions from {len(episode_files)} episodes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_t, frame_tp1, action = self.samples[idx]
        # Stack frames along channel dim: (6, H, W)
        x = torch.cat([frame_t, frame_tp1], dim=0)
        return x, action


class IDM(nn.Module):
    """Small CNN that predicts action from two consecutive frames."""

    def __init__(self, num_actions: int = 10, img_channels: int = 3):
        super().__init__()
        in_ch = img_channels * 2  # two frames stacked
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        h = self.features(x)
        h = h.flatten(1)
        return self.classifier(h)


def train_idm(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_ds = IDMDataset(args.data, split="train")
    val_ds = IDMDataset(args.data, split="val")
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = IDM(num_actions=args.num_actions).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    params = sum(p.numel() for p in model.parameters())
    print(f"IDM params: {params:,}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = 0.0
    for epoch in range(args.epochs):
        # Train
        model.train()
        losses = []
        for x, action in train_dl:
            x, action = x.to(device), action.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, action in val_dl:
                x, action = x.to(device), action.to(device)
                logits = model(x)
                correct += (logits.argmax(1) == action).sum().item()
                total += len(action)

        acc = correct / total if total > 0 else 0
        print(f"Epoch {epoch+1}/{args.epochs}: loss={np.mean(losses):.4f}, val_acc={acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save({"model": model.state_dict(), "acc": acc, "epoch": epoch},
                        out_dir / "best.pt")

    print(f"\nBest val accuracy: {best_acc:.4f}")
    print(f"Saved to {out_dir / 'best.pt'}")
    return best_acc


def eval_idm_on_worldmodel(args):
    """Evaluate IDM accuracy on world-model-generated transitions."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load IDM
    idm = IDM(num_actions=args.num_actions).to(device)
    ckpt = torch.load(args.idm_checkpoint, map_location=device, weights_only=False)
    idm.load_state_dict(ckpt["model"])
    idm.eval()
    print(f"Loaded IDM (val_acc={ckpt['acc']:.4f})")

    # Load world model
    from model import make_denoiser
    wm_ckpt = torch.load(args.wm_checkpoint, map_location=device, weights_only=False)
    wm = make_denoiser(
        num_actions=args.num_actions,
        img_size=args.res,
        num_context_frames=args.num_context,
        model_size=args.model_size,
    ).to(device)
    wm.load_state_dict(wm_ckpt["model"])
    wm.eval()
    print(f"Loaded world model (loss={wm_ckpt['loss']:.4f})")

    # Generate transitions using world model and test IDM
    episode_files = sorted(Path(args.data).glob("episode_*.pt"))
    L = args.num_context

    correct, total = 0, 0
    for ep_idx in range(min(args.num_eval_episodes, len(episode_files))):
        ep = Episode.load(episode_files[ep_idx])
        if len(ep) < L + 20:
            continue

        for t in range(L + 5, min(L + 55, len(ep))):
            context = ep.obs[t - L : t].unsqueeze(0).to(device)
            action = ep.act[t].unsqueeze(0).to(device)

            with torch.no_grad():
                pred_next = wm.sample(context, action, num_steps=args.num_denoise_steps)

            # IDM: predict action from (real_frame_t, predicted_frame_t+1)
            real_frame = ep.obs[t].unsqueeze(0).to(device)  # (1, C, H, W) in [-1,1]
            idm_input = torch.cat([real_frame, pred_next], dim=1)  # (1, 6, H, W)

            with torch.no_grad():
                pred_action = idm(idm_input).argmax(1)

            correct += (pred_action == action).sum().item()
            total += 1

    acc = correct / total if total > 0 else 0
    print(f"\nIDM accuracy on world model transitions: {acc:.4f} ({correct}/{total})")
    print(f"(Random baseline: {1/args.num_actions:.4f})")
    return acc


def main():
    parser = argparse.ArgumentParser(description="Inverse Dynamics Model")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--data", type=str, default="data/episodes")
    train_p.add_argument("--output", type=str, default="experiments/idm")
    train_p.add_argument("--num_actions", type=int, default=10)
    train_p.add_argument("--epochs", type=int, default=20)
    train_p.add_argument("--batch_size", type=int, default=256)
    train_p.add_argument("--lr", type=float, default=1e-3)

    # Eval
    eval_p = subparsers.add_parser("eval")
    eval_p.add_argument("--idm_checkpoint", type=str, required=True)
    eval_p.add_argument("--wm_checkpoint", type=str, required=True)
    eval_p.add_argument("--data", type=str, default="data/episodes")
    eval_p.add_argument("--num_actions", type=int, default=10)
    eval_p.add_argument("--res", type=int, default=84)
    eval_p.add_argument("--num_context", type=int, default=4)
    eval_p.add_argument("--model_size", type=str, default="small")
    eval_p.add_argument("--num_denoise_steps", type=int, default=3)
    eval_p.add_argument("--num_eval_episodes", type=int, default=10)

    args = parser.parse_args()
    if args.command == "train":
        train_idm(args)
    elif args.command == "eval":
        eval_idm_on_worldmodel(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
