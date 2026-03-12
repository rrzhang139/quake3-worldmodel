"""Episode data format compatible with DIAMOND.

Each episode stores:
  obs:   (T+1, C, H, W) float32 in [-1, 1]  (T+1 because we need obs before first action)
  act:   (T,) int64 discrete action indices
  rew:   (T,) float32 rewards
  end:   (T,) bool terminal flags
  trunc: (T,) bool truncation flags
"""

from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np


@dataclass
class Episode:
    obs: torch.FloatTensor    # (T+1, C, H, W) in [-1, 1]
    act: torch.LongTensor     # (T,)
    rew: torch.FloatTensor    # (T,)
    end: torch.BoolTensor     # (T,)
    trunc: torch.BoolTensor   # (T,)

    def __len__(self):
        return self.act.shape[0]

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Store obs as uint8 to save disk space
        obs_uint8 = ((self.obs + 1) / 2 * 255).clamp(0, 255).byte()
        torch.save({
            "obs": obs_uint8,
            "act": self.act,
            "rew": self.rew,
            "end": self.end,
            "trunc": self.trunc,
        }, path)

    @staticmethod
    def load(path: Path) -> "Episode":
        d = torch.load(path, weights_only=True)
        obs = d["obs"].float() / 255 * 2 - 1  # uint8 -> [-1, 1]
        return Episode(
            obs=obs,
            act=d["act"],
            rew=d["rew"],
            end=d["end"],
            trunc=d["trunc"],
        )

    @staticmethod
    def from_numpy(obs_list, act_list, rew_list, end_list, trunc_list) -> "Episode":
        """Create from lists of numpy arrays collected during rollout.

        obs_list: list of (H, W, C) uint8 arrays, length T+1
        act_list: list of int, length T
        rew_list: list of float, length T
        end_list: list of bool, length T
        trunc_list: list of bool, length T
        """
        # (T+1, H, W, C) uint8 -> (T+1, C, H, W) float [-1, 1]
        obs = np.stack(obs_list)
        obs = torch.from_numpy(obs).permute(0, 3, 1, 2).float() / 255 * 2 - 1

        return Episode(
            obs=obs,
            act=torch.tensor(act_list, dtype=torch.long),
            rew=torch.tensor(rew_list, dtype=torch.float32),
            end=torch.tensor(end_list, dtype=torch.bool),
            trunc=torch.tensor(trunc_list, dtype=torch.bool),
        )
