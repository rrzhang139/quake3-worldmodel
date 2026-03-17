"""Train an Inverse Dynamics Model (IDM) on real episode data.

The IDM predicts which action was taken given two consecutive frames.
Used as an eval metric: if the world model generates correct visual effects
for each action, the IDM should predict the action correctly from generated frames.

Data flow:
  1. Load episodes from disk (or HuggingFace)
  2. For each pair (frame_t, frame_t+1, action_t):
     - frame_t and frame_t+1 are uint8 (C, H, W), converted to float [0, 1]
     - Concatenated along channel dim → (2C, H, W) input
     - action_t is the label (int 0-9)
  3. Train a tiny CNN (3 conv layers, ~50K params) with cross-entropy
  4. Save weights to disk (and optionally W&B)
  5. Optionally save the (frame_t, frame_t+1, action) pairs to HuggingFace

Usage:
    # Train on local episodes
    python src/train_idm.py --data data/episodes_mixed --output models/idm.pt

    # Train on HuggingFace dataset
    python src/train_idm.py --hf_dataset rzhang139/vizdoom-episodes --hf_subset episodes_84x84_10k --output models/idm.pt

    # Also upload to W&B and save training pairs to HF
    python src/train_idm.py --data data/episodes_mixed --output models/idm.pt --wandb --save_pairs_hf
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from episode import Episode


class InverseDynamicsModel(nn.Module):
    """Tiny CNN that predicts action from (frame_t, frame_t+1).

    Input: (batch, 2*C, H, W) — two frames concatenated along channel dim
    Output: (batch, num_actions) — logits for each action

    Architecture:
      conv 2C→32 (3x3, stride 2) → ReLU   # downsample 2x, extract low-level features
      conv 32→64 (3x3, stride 2) → ReLU    # downsample 2x, extract mid-level features
      conv 64→64 (3x3, stride 2) → ReLU    # downsample 2x, extract high-level features
      adaptive avg pool → (64,)             # collapse spatial dims
      linear 64→num_actions                 # classify action

    For 84x84 input: 84→42→21→11→pool→64→10
    For 160x120 input: 120x160→60x80→30x40→15x20→pool→64→10
    ~50K parameters. Trains in <1 minute on CPU.
    """

    def __init__(self, num_actions=10, in_channels=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        return self.net(x)


def load_episodes(data_dir, max_episodes=None):
    """Load episodes from a directory of .pt files."""
    episode_files = sorted(Path(data_dir).glob("episode_*.pt"))
    if max_episodes:
        episode_files = episode_files[:max_episodes]
    episodes = []
    for f in episode_files:
        episodes.append(Episode.load(f))
    print(f"Loaded {len(episodes)} episodes from {data_dir}")
    return episodes


def build_training_pairs(episodes, num_samples=50000):
    """Extract (frame_t, frame_t+1, action) pairs from episodes.

    Data flow per sample:
      Pick random episode → pick random timestep t
      → obs[t] is (C, H, W) float [-1, 1] → shift to [0, 1]
      → obs[t+1] is (C, H, W) float [-1, 1] → shift to [0, 1]
      → concat along channel dim → (2C, H, W)
      → action = act[t] (int 0-9)
    """
    pairs_x = []
    pairs_y = []

    for _ in range(num_samples):
        ep = episodes[np.random.randint(len(episodes))]
        t = np.random.randint(0, len(ep))
        # obs is float [-1, 1], shift to [0, 1] for IDM
        frame_t = (ep.obs[t] + 1) / 2
        frame_tp1 = (ep.obs[t + 1] + 1) / 2
        pairs_x.append(torch.cat([frame_t, frame_tp1], dim=0))  # (2C, H, W)
        pairs_y.append(ep.act[t])

    X = torch.stack(pairs_x)   # (N, 2C, H, W)
    Y = torch.stack(pairs_y)   # (N,)
    print(f"Built {len(X)} training pairs, input shape: {X.shape}")
    return X, Y


def train(X, Y, num_actions=10, epochs=10, batch_size=256, lr=1e-3, device="cpu"):
    """Train the IDM on prepared pairs.

    Data flow per batch:
      X_batch (bs, 2C, H, W) float [0, 1]
        → IDM forward → logits (bs, 10)
        → cross-entropy with Y_batch (bs,) int
        → backward → Adam step
    """
    in_channels = X.shape[1]
    idm = InverseDynamicsModel(num_actions, in_channels).to(device)
    optimizer = torch.optim.Adam(idm.parameters(), lr=lr)

    X = X.to(device)
    Y = Y.to(device)

    idm.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        total_loss = 0
        correct = 0

        for i in range(0, len(X), batch_size):
            batch_x = X[perm[i:i+batch_size]]
            batch_y = Y[perm[i:i+batch_size]]

            logits = idm(batch_x)
            loss = F.cross_entropy(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            correct += (logits.argmax(1) == batch_y).sum().item()

        acc = correct / len(X)
        avg_loss = total_loss / len(X)
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, accuracy={acc:.1%}")

    idm.eval()
    return idm


def main():
    parser = argparse.ArgumentParser(description="Train Inverse Dynamics Model")
    parser.add_argument("--data", type=str, default="data/episodes_mixed",
                        help="Directory with episode .pt files")
    parser.add_argument("--output", type=str, default=None,
                        help="Where to save IDM weights. Default: models/idm_<resolution>.pt")
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of (frame_t, frame_t+1, action) pairs to train on")
    parser.add_argument("--max_episodes", type=int, default=None,
                        help="Max episodes to load (None = all)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wandb", action="store_true", help="Upload weights to W&B")
    parser.add_argument("--save_pairs_hf", action="store_true",
                        help="Save training pairs to HuggingFace dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    episodes = load_episodes(args.data, args.max_episodes)

    # Build training pairs
    X, Y = build_training_pairs(episodes, args.num_samples)

    # Detect resolution from first sample: X is (N, 2C, H, W)
    H, W = X.shape[2], X.shape[3]
    res_label = f"{W}x{H}"  # e.g. "84x84" or "160x120"
    print(f"Resolution detected: {res_label}")

    # Train
    print(f"\nTraining IDM ({args.epochs} epochs, {args.num_samples} samples)...")
    idm = train(X, Y, args.num_actions, args.epochs, args.batch_size, args.lr, device)

    # Save weights with resolution in filename
    out_path = Path(args.output) if args.output else Path(f"models/idm_{res_label}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": idm.state_dict(),
        "num_actions": args.num_actions,
        "in_channels": X.shape[1],
        "resolution": res_label,
        "height": H,
        "width": W,
    }, out_path)
    print(f"\nSaved IDM to {out_path} (resolution: {res_label})")

    # Upload to W&B
    if args.wandb:
        import wandb
        import os
        wandb.login(key=os.environ.get("WANDB_API_KEY", ""))
        run = wandb.init(project="quake3-worldmodel", entity="rzhang139",
                         job_type="idm", name="idm-training")
        art = wandb.Artifact("quake3-idm", type="model",
                             description=f"IDM trained on {len(episodes)} episodes, "
                             f"{args.num_samples} pairs, {args.epochs} epochs")
        art.add_file(str(out_path))
        run.log_artifact(art)
        run.finish()
        print("Uploaded IDM to W&B")

    # Save training pairs to HuggingFace
    if args.save_pairs_hf:
        pairs_dir = Path("data/idm_pairs")
        pairs_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"X": X.cpu(), "Y": Y.cpu()}, pairs_dir / "pairs.pt")
        print(f"Saved {len(X)} pairs to {pairs_dir}")

        from huggingface_hub import HfApi
        import os
        api = HfApi(token=os.environ.get("HF_TOKEN", ""))
        api.upload_folder(
            folder_path=str(pairs_dir),
            repo_id="rzhang139/vizdoom-episodes",
            repo_type="dataset",
            path_in_repo="idm_pairs",
            commit_message=f"IDM training pairs ({len(X)} samples)",
        )
        print("Uploaded IDM pairs to HuggingFace")


if __name__ == "__main__":
    main()
