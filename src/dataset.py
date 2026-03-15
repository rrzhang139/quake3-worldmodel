"""PyTorch dataset for world model training.

Loads episodes and samples fixed-length segments for training the denoiser.
Each segment: L context frames + 1 target frame, with the corresponding action.

Supports three modes:
  - "ram" (default): Flattens all obs into a single uint8 ByteTensor in RAM.
    Zero disk IO during training. ~42GB for 2M frames at 84x84x3.
    Fork-safe: torch.Tensor uses torch.Storage (shared memory), NOT Python
    refcounted objects, so forked DataLoader workers share memory without
    copy-on-write duplication.
  - "episodes": Loads Episode objects into RAM (legacy, higher memory due to
    float32 obs per episode). Works for small datasets (<1K episodes).
  - "streaming": LRU cache of episodes loaded from disk on demand.
    Disk IO bound -- causes 0% GPU utilization on large datasets. Avoid.
"""

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader

from episode import Episode


class WorldModelDataset(Dataset):
    """Samples (context_obs, action, target_obs) from stored episodes.

    context_obs: (L, C, H, W)  -- past L frames, float32 in [-1, 1]
    action:      scalar long    -- action taken after last context frame
    target_obs:  (C, H, W)     -- next frame to predict, float32 in [-1, 1]
    """

    def __init__(
        self,
        episode_dir: str,
        num_context_frames: int = 4,
        mode: str = "ram",
        cache_size: int = 500,
    ):
        self.episode_dir = Path(episode_dir)
        self.num_context = num_context_frames
        self.mode = mode

        episode_files = sorted(self.episode_dir.glob("episode_*.pt"))
        if not episode_files:
            raise FileNotFoundError(f"No episodes found in {episode_dir}")

        if mode == "ram":
            self._init_ram(episode_files)
        elif mode == "episodes":
            self._init_episodes(episode_files)
        elif mode == "streaming":
            self._init_streaming(episode_files, cache_size)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'ram', 'episodes', or 'streaming'.")

    # ------------------------------------------------------------------
    # Mode: ram -- flat uint8 tensors, zero-copy fork-safe
    # ------------------------------------------------------------------

    def _init_ram(self, episode_files):
        """Load all episodes, flatten into contiguous uint8 ByteTensor.

        Memory layout:
          self._obs: (total_frames, C, H, W) uint8   -- ALL frames from all episodes
          self._act: (total_actions,) int64            -- ALL actions
          self._index: list of (global_frame_idx_of_t,) for each valid sample

        A sample at index i corresponds to:
          context = _obs[t - L : t]    (L frames)
          target  = _obs[t]            (1 frame)
          action  = _act[t - 1]        (maps to _act via offset)

        To handle episode boundaries, we store per-episode offsets and only
        index within episodes (never cross episode boundaries).
        """
        all_obs = []   # list of uint8 tensors (T+1, C, H, W)
        all_act = []   # list of int64 tensors (T,)
        all_end = []   # list of bool tensors (T,)
        ep_lengths = []  # T for each episode (number of actions)

        print(f"Loading {len(episode_files)} episodes into RAM (uint8)...")
        for i, f in enumerate(episode_files):
            d = torch.load(f, weights_only=True)
            obs_uint8 = d["obs"]  # (T+1, C, H, W) uint8
            act = d["act"]        # (T,) int64
            end = d["end"]        # (T,) bool

            T = act.shape[0]
            if T < self.num_context:
                continue

            all_obs.append(obs_uint8)
            all_act.append(act)
            all_end.append(end)
            ep_lengths.append(T)

            if (i + 1) % 2000 == 0:
                print(f"  loaded {i + 1}/{len(episode_files)} episodes...")

        num_episodes = len(ep_lengths)

        # Concatenate into flat tensors
        self._obs = torch.cat(all_obs, dim=0)  # (total_frames, C, H, W) uint8
        self._act = torch.cat(all_act, dim=0)  # (total_actions,) int64
        all_end_cat = torch.cat(all_end, dim=0)  # (total_actions,) bool

        # Free the lists
        del all_obs, all_act, all_end

        # Build index: for each episode, compute the global offset of frame 0
        # Episode i's obs starts at obs_offsets[i], acts start at act_offsets[i]
        obs_offsets = []  # cumulative obs offset (T+1 per episode)
        act_offsets = []  # cumulative act offset (T per episode)
        obs_off = 0
        act_off = 0
        for T in ep_lengths:
            obs_offsets.append(obs_off)
            act_offsets.append(act_off)
            obs_off += T + 1
            act_off += T

        # Build flat sample index
        # Each sample is identified by (obs_global_t, act_global_t_minus_1)
        # where obs_global_t is the global index of the target frame
        self.index = []
        for ep_i in range(num_episodes):
            T = ep_lengths[ep_i]
            o_off = obs_offsets[ep_i]
            a_off = act_offsets[ep_i]

            for t in range(self.num_context, T + 1):
                # t is the local target frame index within the episode
                # Check if previous step was terminal (skip if so)
                if t > 0 and t <= T and all_end_cat[a_off + t - 1]:
                    continue
                # Store (global_obs_index_of_target, global_act_index_of_action)
                self.index.append((o_off + t, a_off + t - 1))

        del all_end_cat

        obs_gb = self._obs.numel() / 1e9
        print(
            f"Loaded {num_episodes} episodes into RAM: "
            f"{len(self.index)} samples, "
            f"{self._obs.shape[0]} frames, "
            f"{obs_gb:.1f} GB (uint8)"
        )

    # ------------------------------------------------------------------
    # Mode: episodes -- legacy, loads Episode objects into RAM
    # ------------------------------------------------------------------

    def _init_episodes(self, episode_files):
        """Legacy mode: load all episodes as Episode objects (float32 obs)."""
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

        print(f"Loaded {len(self.episodes)} episodes, {len(self.index)} samples (episodes mode)")

    # ------------------------------------------------------------------
    # Mode: streaming -- LRU cache, disk IO on demand
    # ------------------------------------------------------------------

    def _init_streaming(self, episode_files, cache_size):
        """Streaming mode: load episodes from disk on demand with LRU cache."""
        self.episode_files = episode_files
        self.episodes = None
        self._cache = {}
        self._cache_order = []
        self._cache_size = cache_size

        self.index = []
        for ep_idx, f in enumerate(episode_files):
            ep = Episode.load(f)
            ep_len = len(ep)
            if ep_len >= self.num_context:
                for t in range(self.num_context, ep_len + 1):
                    if t > 0 and t <= ep_len and ep.end[t - 1]:
                        continue
                    self.index.append((ep_idx, t))
            del ep

        print(
            f"Indexed {len(episode_files)} episodes, {len(self.index)} samples "
            f"(streaming mode, cache={cache_size})"
        )

    # ------------------------------------------------------------------
    # Shared interface
    # ------------------------------------------------------------------

    def _get_episode(self, ep_idx):
        """Get episode, using LRU cache in streaming mode."""
        if self.mode != "streaming":
            return self.episodes[ep_idx]

        if ep_idx in self._cache:
            return self._cache[ep_idx]

        ep = Episode.load(self.episode_files[ep_idx])
        self._cache[ep_idx] = ep
        self._cache_order.append(ep_idx)

        while len(self._cache_order) > self._cache_size:
            old = self._cache_order.pop(0)
            self._cache.pop(old, None)

        return ep

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if self.mode == "ram":
            return self._getitem_ram(idx)
        else:
            return self._getitem_episode(idx)

    def _getitem_ram(self, idx):
        """Fast path: index directly into flat uint8 tensors."""
        obs_t, act_idx = self.index[idx]

        # Slice uint8 obs, convert to float32 [-1, 1] on the fly
        # This conversion is ~0.01ms for 5 frames of 84x84x3 -- negligible
        context_uint8 = self._obs[obs_t - self.num_context : obs_t]  # (L, C, H, W)
        target_uint8 = self._obs[obs_t]                               # (C, H, W)

        context = context_uint8.float().div_(255).mul_(2).sub_(1)
        target = target_uint8.float().div_(255).mul_(2).sub_(1)
        action = self._act[act_idx]

        return {
            "context": context,    # (L, C, H, W) float32
            "target": target,      # (C, H, W) float32
            "action": action,      # scalar long
        }

    def _getitem_episode(self, idx):
        """Legacy path: index into Episode objects."""
        ep_idx, t = self.index[idx]
        ep = self._get_episode(ep_idx)

        context = ep.obs[t - self.num_context : t]  # (L, C, H, W)
        target = ep.obs[t]                           # (C, H, W)
        action = ep.act[t - 1]                       # scalar

        return {
            "context": context,
            "target": target,
            "action": action,
        }


def make_dataloader(
    episode_dir: str,
    batch_size: int = 32,
    num_context_frames: int = 4,
    num_workers: int = 0,
    shuffle: bool = True,
    mode: str = "ram",
) -> DataLoader:
    """Create DataLoader for world model training.

    Args:
        mode: "ram" (recommended for large datasets that fit in RAM),
              "episodes" (legacy, loads Episode objects),
              "streaming" (disk IO on demand, slow).
    """
    dataset = WorldModelDataset(episode_dir, num_context_frames, mode=mode)

    # For ram mode with num_workers > 0, persistent_workers avoids
    # re-forking (and re-sharing) the tensor each epoch
    use_persistent = num_workers > 0 and mode == "ram"

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=use_persistent,
    )


if __name__ == "__main__":
    import sys
    import time

    episode_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/episodes"
    mode = sys.argv[2] if len(sys.argv) > 2 else "ram"

    print(f"Testing mode={mode}")
    t0 = time.time()
    dl = make_dataloader(episode_dir, batch_size=64, num_context_frames=4, mode=mode, num_workers=2)
    t_load = time.time() - t0
    print(f"Load time: {t_load:.1f}s")

    # Benchmark: iterate one full epoch
    t0 = time.time()
    n_batches = 0
    for batch in dl:
        n_batches += 1
        if n_batches >= 100:
            break
    t_iter = time.time() - t0
    print(f"100 batches in {t_iter:.2f}s ({100/t_iter:.0f} batches/sec)")
    print(f"  context: {batch['context'].shape}")
    print(f"  target:  {batch['target'].shape}")
    print(f"  action:  {batch['action'].shape}")
