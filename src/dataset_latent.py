"""Dataset for latent-space world model training.

Loads pre-encoded VAE latent episodes (from encode_latents.py).
Each sample: 4 context latent frames + 1 target latent frame + action.

Latent frames are (4, 15, 20) float32 — already scaled by 0.18215.
No uint8→float conversion needed (latents are always float).
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader


class LatentDataset(Dataset):
    """Flat RAM dataset for pre-encoded VAE latents.

    Same architecture as WorldModelDataset RAM mode, but for float32 latents
    instead of uint8 pixels. Each frame is (4, 15, 20) = 1,200 floats = 4.8KB.
    5K episodes × ~200 frames = ~1M frames × 4.8KB = ~4.8GB RAM (vs 19GB for pixels).
    """

    def __init__(self, episode_dir: str, num_context_frames: int = 4):
        self.num_context = num_context_frames
        episode_dir = Path(episode_dir)

        episode_files = sorted(episode_dir.glob("episode_*.pt"))
        if not episode_files:
            raise FileNotFoundError(f"No episodes found in {episode_dir}")

        all_obs = []
        all_act = []
        all_end = []
        ep_lengths = []

        print(f"Loading {len(episode_files)} latent episodes into RAM...")
        for i, f in enumerate(episode_files):
            d = torch.load(f, weights_only=True)
            obs = d["obs_latent"]  # (T+1, 4, H_lat, W_lat) float32
            act = d["act"]         # (T,) int64
            end = d["end"]         # (T,) bool

            T = act.shape[0]
            if T < self.num_context:
                continue

            all_obs.append(obs)
            all_act.append(act)
            all_end.append(end)
            ep_lengths.append(T)

            if (i + 1) % 1000 == 0:
                print(f"  loaded {i+1}/{len(episode_files)}...")

        num_episodes = len(ep_lengths)
        self._obs = torch.cat(all_obs, dim=0)  # (total_frames, 4, H, W) float32
        self._act = torch.cat(all_act, dim=0)
        all_end_cat = torch.cat(all_end, dim=0)
        del all_obs, all_act, all_end

        # Build index (same logic as RAM pixel dataset)
        obs_offsets = []
        act_offsets = []
        obs_off = 0
        act_off = 0
        for T in ep_lengths:
            obs_offsets.append(obs_off)
            act_offsets.append(act_off)
            obs_off += T + 1
            act_off += T

        self.index = []
        for ep_i in range(num_episodes):
            T = ep_lengths[ep_i]
            o_off = obs_offsets[ep_i]
            a_off = act_offsets[ep_i]
            for t in range(self.num_context, T + 1):
                if t > 0 and t <= T and all_end_cat[a_off + t - 1]:
                    continue
                self.index.append((o_off + t, a_off + t - 1))

        del all_end_cat
        obs_gb = self._obs.numel() * 4 / 1e9  # float32 = 4 bytes
        print(f"Loaded {num_episodes} episodes: {len(self.index)} samples, "
              f"{self._obs.shape[0]} frames, {obs_gb:.1f} GB, "
              f"latent shape: {self._obs.shape[1:]}")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        obs_t, act_idx = self.index[idx]
        context = self._obs[obs_t - self.num_context : obs_t]  # (L, 4, H, W)
        target = self._obs[obs_t]                                # (4, H, W)
        action = self._act[act_idx]
        return {"context": context, "target": target, "action": action}


def make_latent_dataloader(
    episode_dir: str,
    batch_size: int = 64,
    num_context_frames: int = 4,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    dataset = LatentDataset(episode_dir, num_context_frames)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
