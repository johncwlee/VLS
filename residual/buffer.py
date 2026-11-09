"""Chunk-level replay buffer on disk (np.memmap), with spec §9's two pools and sample-time rewards.

One row per executed chunk; episodes are stored contiguously, so the next observation of row i is row i + 1
(terminal rows have no successor and bootstrap with mask 0). Rows store raw facts (steered, terminal, success)
and rewards are computed at sample time by `chunk_reward`, so reward hyperparameters can change without
re-collecting. Pools: protected = offline (Stage 1) episodes + every steered online chunk; online = every
online chunk. Held-out episodes are never sampled for training.
"""

import json
import os
from pathlib import Path

import numpy as np


def chunk_reward(steered: np.ndarray, terminal: np.ndarray, success: np.ndarray, kappa: float,
                 failure_penalty: float | None, t_max: int | np.ndarray, normalize: bool
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Spec §7.1 per chunk: -1, -kappa if steered, -C on failure/timeout (C defaults to 2 * t_max); success
    is terminal with no bonus. t_max (max episode chunks) is a scalar or one value per row. Returns (reward,
    bootstrap mask)."""
    c_fail = 2.0 * t_max if failure_penalty is None else float(failure_penalty)
    reward = -1.0 - kappa * steered - c_fail * (terminal & ~success)
    if normalize:
        reward = reward / t_max
    return reward.astype(np.float32), (~terminal).astype(np.float32)


def field_specs(image_shape: tuple[int, ...], state_dim: int, exec_horizon: int, action_dim: int, horizon: int,
                pad_dim: int, prefix_dim: int, ae_dim: int) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {
        "image": ("uint8", image_shape), "image2": ("uint8", image_shape),  # env-preprocessed (flipped) cameras
        "state": ("float32", (state_dim,)),       # raw proprio, re-normalized by the policy preprocessor
        "state_norm": ("float32", (state_dim,)),  # as the policy saw it (critic features without a π0.5 pass)
        "task": ("int16", ()), "episode": ("int32", ()), "step": ("int16", ()),
        "a_exec": ("float32", (exec_horizon, action_dim)),  # normalized, clipped, edited: the TD action
        "a_base": ("float32", (exec_horizon, action_dim)),  # π0.5 alone from the same noise (diagnostic only)
        "noise": ("float16", (horizon, pad_dim)),
        "h_prefix": ("float16", (prefix_dim,)), "latent": ("float16", (ae_dim,)),
        "steered": ("bool", ()), "trigger": ("uint8", ()), "terminal": ("bool", ()), "success": ("bool", ()),
        "protected": ("bool", ()), "online": ("bool", ()), "heldout": ("bool", ()),
    }


class ReplayBuffer:
    def __init__(self, directory: Path, capacity: int, specs: dict, readonly: bool = False):
        """readonly: a view of a buffer another process writes (multi-GPU ranks > 0); `reload` picks up its episodes."""
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.dir / "meta.json"
        resume = self.meta_path.exists()
        if resume:
            meta = json.loads(self.meta_path.read_text())
            if meta["capacity"] != capacity or meta["specs"] != {k: [d, list(s)] for k, (d, s) in specs.items()}:
                raise ValueError(f"Buffer at {self.dir} was created with a different capacity or layout")
            self.size, self.episodes = meta["size"], meta["episodes"]
        else:
            self.size, self.episodes = 0, []
        self.capacity, self.specs = capacity, specs
        self.data = {k: np.memmap(self.dir / f"{k}.bin", dtype=d, mode="r" if readonly else "r+" if resume else "w+", shape=(capacity, *s))
                     for k, (d, s) in specs.items()}
        self._refresh_index()

    def reload(self) -> None:
        meta = json.loads(self.meta_path.read_text())
        self.size, self.episodes = meta["size"], meta["episodes"]
        self._refresh_index()

    def _refresh_index(self) -> None:
        n, d = self.size, self.data
        trainable = ~d["heldout"][:n]
        self.protected_idx = np.flatnonzero(d["protected"][:n] & trainable)
        self.online_idx = np.flatnonzero(d["online"][:n] & trainable)

    def add_episode(self, chunks: dict[str, np.ndarray], meta: dict, online: bool, heldout: bool) -> None:
        """Append one finished episode (dict of [T, ...] arrays) and persist the index atomically."""
        length = len(chunks["a_exec"])
        if self.size + length > self.capacity:
            # ponytail: no ring overwrite (protected data must survive); raise capacity if a run outgrows it
            raise RuntimeError(f"Replay buffer full ({self.capacity} chunks); raise train.buffer_capacity")
        rows = slice(self.size, self.size + length)
        for k, arr in chunks.items():
            self.data[k][rows] = arr
        steered = np.asarray(chunks["steered"], dtype=bool)
        self.data["online"][rows] = online
        self.data["protected"][rows] = steered | (not online)
        self.data["heldout"][rows] = heldout
        self.data["episode"][rows] = len(self.episodes)
        for arr in self.data.values():
            arr.flush()
        self.episodes.append({**meta, "id": len(self.episodes), "start": self.size, "length": length,
                              "online": online, "heldout": heldout})
        self.size += length
        tmp = self.meta_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"capacity": self.capacity, "size": self.size, "episodes": self.episodes,
                                   "specs": {k: [d, list(s)] for k, (d, s) in self.specs.items()}}))
        os.replace(tmp, self.meta_path)
        self._refresh_index()

    def sample(self, batch_size: int, offline_ratio: float, rng: np.random.Generator) -> np.ndarray:
        """Spec §9: offline_ratio of the batch from the protected pool, the rest from the online pool."""
        if len(self.online_idx) == 0 or len(self.protected_idx) == 0:
            pool = self.protected_idx if len(self.online_idx) == 0 else self.online_idx
            if len(pool) == 0:
                raise RuntimeError("Replay buffer has no trainable chunks")
            return rng.choice(pool, batch_size)
        n_prot = int(round(batch_size * offline_ratio))
        return np.concatenate([rng.choice(self.protected_idx, n_prot), rng.choice(self.online_idx, batch_size - n_prot)])

    def next_index(self, idx: np.ndarray) -> np.ndarray:
        return np.where(self.data["terminal"][idx], idx, idx + 1)

    def get(self, idx: np.ndarray, keys: list[str]) -> dict[str, np.ndarray]:
        order = np.argsort(idx)  # sorted reads are friendlier to the memmap; restore the caller's order after
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return {k: np.asarray(self.data[k][idx[order]])[inverse] for k in keys}

    def episode_count(self, **match) -> int:
        return sum(all(ep.get(k) == v for k, v in match.items()) for ep in self.episodes)
