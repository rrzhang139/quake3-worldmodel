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

    Supports two modes:
    - streaming=False (default for small datasets): loads all episodes into RAM
    - streaming=True (for large datasets): loads episodes from disk on demand with LRU cache
    """

    def __init__(self, episode_dir: str, num_context_frames: int = 4, streaming: bool = False, cache_size: int = 500):
        self.episode_dir = Path(episode_dir)
        self.num_context = num_context_frames
        self.streaming = streaming

        episode_files = sorted(self.episode_dir.glob("episode_*.pt"))
        if not episode_files:
            raise FileNotFoundError(f"No episodes found in {episode_dir}")

        if streaming:
            # Streaming mode: only scan episode lengths, don't load obs
            self.episode_files = episode_files
            self.episodes = None
            self._cache = {}
            self._cache_order = []
            self._cache_size = cache_size

            # Build index by scanning episode metadata (fast, low memory)
            self.index = []
            for ep_idx, f in enumerate(episode_files):
                ep = Episode.load(f)
                ep_len = len(ep)
                if ep_len >= self.num_context:
                    for t in range(self.num_context, ep_len + 1):
                        if t > 0 and t <= ep_len and ep.end[t - 1]:
                            continue
                        self.index.append((ep_idx, t))
                # Don't keep episode in memory
                del ep

            print(f"Indexed {len(episode_files)} episodes, {len(self.index)} samples (streaming mode, cache={cache_size})")
        else:
            # In-memory mode: load all episodes
            self.episodes = []
            for f in episode_files:
                ep = Episode.load(f)
                if len(ep) >= self.num_context:
                    self.episodes.append(ep)

            self.index = []
            for ep_idx, ep in enumerate(self.episodes):
                for t in range(self.num_context, len(ep) + 1):
                    if t > 0 and t <= len(ep) and ep.end[t - 1]:
                        continue
                    self.index.append((ep_idx, t))

            print(f"Loaded {len(self.episodes)} episodes, {len(self.index)} samples")

    def _get_episode(self, ep_idx):
        """Get episode, using LRU cache in streaming mode."""
        if not self.streaming:
            return self.episodes[ep_idx]

        if ep_idx in self._cache:
            return self._cache[ep_idx]

        ep = Episode.load(self.episode_files[ep_idx])
        self._cache[ep_idx] = ep
        self._cache_order.append(ep_idx)

        # Evict oldest if cache full
        while len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return ep

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        ep_idx, t = self.index[idx]
        ep = self._get_episode(ep_idx)

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
    streaming: bool = False,
) -> DataLoader:
    dataset = WorldModelDataset(episode_dir, num_context_frames, streaming=streaming)
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
