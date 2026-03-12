"""PyTorch dataset for world model training.

Loads episodes and samples fixed-length segments for training the denoiser.
Each segment: L context frames + 1 target frame, with the corresponding action.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from episode import Episode


class WorldModelDataset(Dataset):
    """Samples (context_obs, action, target_obs) from stored episodes.

    context_obs: (L, C, H, W)  -- past L frames
    action:      (1,) long      -- action taken after last context frame
    target_obs:  (C, H, W)      -- next frame to predict
    """

    def __init__(self, episode_dir: str, num_context_frames: int = 4):
        self.episode_dir = Path(episode_dir)
        self.num_context = num_context_frames

        # Load all episodes into memory
        self.episodes = []
        episode_files = sorted(self.episode_dir.glob("episode_*.pt"))
        if not episode_files:
            raise FileNotFoundError(f"No episodes found in {episode_dir}")

        for f in episode_files:
            ep = Episode.load(f)
            if len(ep) >= self.num_context:
                self.episodes.append(ep)

        # Build index: (episode_idx, timestep) for valid samples
        self.index = []
        for ep_idx, ep in enumerate(self.episodes):
            # timestep t means: context = obs[t-L:t], action = act[t-1], target = obs[t]
            for t in range(self.num_context, len(ep) + 1):
                # Skip if the target frame is after a terminal
                if t > 0 and t <= len(ep) and ep.end[t - 1]:
                    continue
                self.index.append((ep_idx, t))

        print(f"Loaded {len(self.episodes)} episodes, {len(self.index)} samples")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        ep = self.episodes[ep_idx]

        # Context: L frames before timestep t
        context = ep.obs[t - self.num_context : t]  # (L, C, H, W)

        # Target: frame at timestep t
        target = ep.obs[t]  # (C, H, W)

        # Action: the action taken at timestep t-1 (leading to the target)
        action = ep.act[t - 1]  # scalar

        return {
            "context": context,    # (L, C, H, W)
            "target": target,      # (C, H, W)
            "action": action,      # scalar long
        }


def make_dataloader(
    episode_dir: str,
    batch_size: int = 32,
    num_context_frames: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
) -> DataLoader:
    dataset = WorldModelDataset(episode_dir, num_context_frames)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    # Quick test
    dl = make_dataloader("../data/episodes", batch_size=4, num_context_frames=4)
    batch = next(iter(dl))
    print("context:", batch["context"].shape)   # (B, L, C, H, W)
    print("target:", batch["target"].shape)     # (B, C, H, W)
    print("action:", batch["action"].shape)     # (B,)
    print("action values:", batch["action"].tolist())
