import bisect
import json
import random
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from config import SPLITS
from utils import sha256_file


def split_dir(cache_dir, fragment_length: int, split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    return Path(cache_dir) / "dataset" / f"{fragment_length}bp" / split


def load_manifest(cache_dir, fragment_length: int, split: str) -> dict:
    path = split_dir(cache_dir, fragment_length, split) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing preprocessed manifest: {path}\n"
            f"Run: python prepare_data.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


class ShardedSequenceDataset(Dataset):
    """
    Map-style dataset backed by .npy memmaps.

    Global block shuffling preserves full coverage. The LRU cache bounds the
    number of open memory-mapped shards in each worker.
    """

    def __init__(self, cache_dir, fragment_length: int, split: str, max_open_shards: int = 64, verify_hashes: bool = False):
        self.fragment_length = int(fragment_length)
        self.split = split
        self.root = split_dir(cache_dir, fragment_length, split)
        self.manifest = load_manifest(cache_dir, fragment_length, split)
        self.shards = self.manifest["shards"]
        self.max_open_shards = int(max_open_shards)
        if self.max_open_shards <= 0:
            raise ValueError("max_open_shards must be positive")
        self.fingerprint = sha256_file(self.root / "manifest.json")
        if self.manifest["fragment_length"] != self.fragment_length or self.manifest["split"] != split:
            raise ValueError("Manifest length/split mismatch")
        for shard in self.shards:
            if int(shard["n"]) <= 0:
                raise ValueError("Shard must not be empty")
            for key in ("seq_file", "label_file"):
                p = self.root / shard[key]
                if p.parent != self.root or not p.is_file():
                    raise ValueError(f"Missing or invalid shard path: {p}")
                if verify_hashes and sha256_file(p) != shard[key + "_sha256"]:
                    raise ValueError(f"Shard checksum mismatch: {p}")

        self.counts = [int(s["n"]) for s in self.shards]
        self.cum_ends = np.cumsum(self.counts).tolist()
        self.starts = [0] + self.cum_ends[:-1]
        self.total = int(sum(self.counts))
        if not self.total or self.total != int(self.manifest["count"]):
            raise ValueError("Empty or inconsistent split manifest")
        classes = self.manifest["class_counts"]
        if set(classes) != {"host_0", "virus_1"} or sum(classes.values()) != self.total:
            raise ValueError("Inconsistent class counts")
        meta = self.root / self.manifest["metadata_file"]
        if meta.parent != self.root or not meta.is_file():
            raise ValueError("Missing metadata file")
        if verify_hashes and sha256_file(meta) != self.manifest["metadata_sha256"]:
            raise ValueError("Metadata checksum mismatch")

        self._seq_cache = OrderedDict()
        self._label_cache = OrderedDict()

    def __len__(self):
        return self.total

    def _get_maps(self, shard_idx: int):
        if shard_idx in self._seq_cache:
            seq = self._seq_cache.pop(shard_idx)
            lab = self._label_cache.pop(shard_idx)
            self._seq_cache[shard_idx] = seq
            self._label_cache[shard_idx] = lab
            return seq, lab

        meta = self.shards[shard_idx]
        seq = np.load(self.root / meta["seq_file"], mmap_mode="r")
        lab = np.load(self.root / meta["label_file"], mmap_mode="r")

        if seq.shape != (meta["n"], self.fragment_length) or lab.shape != (meta["n"],):
            raise ValueError("Shard shape mismatch")
        if seq.dtype != np.uint8 or lab.dtype != np.uint8:
            raise ValueError("Shard dtype must be uint8")
        self._seq_cache[shard_idx] = seq
        self._label_cache[shard_idx] = lab
        while len(self._seq_cache) > self.max_open_shards:
            self._seq_cache.popitem(last=False)
            self._label_cache.popitem(last=False)
        return seq, lab

    def _locate(self, index: int):
        if index < 0:
            index += self.total
        if index < 0 or index >= self.total:
            raise IndexError(index)
        shard_idx = bisect.bisect_right(self.cum_ends, index)
        local = index - self.starts[shard_idx]
        return shard_idx, local

    def __getitem__(self, index: int):
        shard_idx, local = self._locate(int(index))
        seq_map, lab_map = self._get_maps(shard_idx)

        # Copy one row out of the read-only memmap so PyTorch receives writable memory.
        seq = torch.from_numpy(np.array(seq_map[local], dtype=np.uint8, copy=True))
        label = int(lab_map[local])
        if label not in (0, 1) or bool((seq > 4).any()):
            raise ValueError("Invalid cached token or label")
        return seq, label

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_seq_cache"] = OrderedDict()
        state["_label_cache"] = OrderedDict()
        return state


class DeterministicBlockShuffleSampler(Sampler):
    # Deterministic full-coverage sampler with GLOBAL block mixing.
    # This changes only sample order inside the fixed training split.

    def __init__(
        self,
        dataset: ShardedSequenceDataset,
        seed: int,
        block_size: int = 64,
    ):
        if dataset.split != "train":
            raise ValueError("DeterministicBlockShuffleSampler is for train only.")
        self.dataset = dataset
        self.seed = int(seed)
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        self.epoch = 0

        parts = []
        for shard_idx, n in enumerate(self.dataset.counts):
            starts = np.arange(0, n, self.block_size, dtype=np.int64)
            ends = np.minimum(starts + self.block_size, n)
            shard_col = np.full_like(starts, shard_idx)
            parts.append(np.stack([shard_col, starts, ends], axis=1))
        self.blocks = np.concatenate(parts, axis=0)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _mix_seed(seed: int, epoch: int, shard_idx: int, block_start: int = 0) -> int:
        x = (
            seed * 1_000_003
            + epoch * 97_003
            + shard_idx * 9_973
            + block_start * 433
            + 17
        )
        return int(x % (2**63 - 1))

    def __iter__(self):
        rng = np.random.default_rng(self._mix_seed(self.seed, self.epoch, 0, 0))
        order = np.arange(len(self.blocks), dtype=np.int64)
        rng.shuffle(order)

        for block_id in order.tolist():
            shard_idx, local_start, local_end = self.blocks[block_id].tolist()
            local_indices = np.arange(local_start, local_end, dtype=np.int64)

            rng_inside = np.random.default_rng(
                self._mix_seed(
                    self.seed,
                    self.epoch,
                    int(shard_idx),
                    int(local_start),
                )
            )
            rng_inside.shuffle(local_indices)

            global_start = self.dataset.starts[int(shard_idx)]
            for local_idx in local_indices.tolist():
                yield global_start + local_idx


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
