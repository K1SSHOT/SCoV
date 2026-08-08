from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numba import njit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


_BASE_LOOKUP = np.full(256, 4, dtype=np.uint8)
for _base, _index in (
    ("A", 0), ("C", 1), ("G", 2), ("T", 3),
    ("a", 0), ("c", 1), ("g", 2), ("t", 3),
):
    _BASE_LOOKUP[ord(_base)] = _index


def encode_sequence(sequence: str, length: int) -> np.ndarray:
    raw = np.frombuffer(
        str(sequence)[:length].encode("ascii", errors="replace"),
        dtype=np.uint8,
    )
    encoded = np.full(length, 4, dtype=np.uint8)
    n = min(raw.size, length)
    encoded[:n] = _BASE_LOOKUP[raw[:n]]
    return encoded


@njit(cache=True)
def reverse_complement(sequence: np.ndarray) -> np.ndarray:
    output = np.empty_like(sequence)
    n = len(sequence)
    for index in range(n):
        base = sequence[n - 1 - index]
        output[index] = 3 - base if base < 4 else 4
    return output


@njit(cache=True)
def six_frame_codon_ids(sequence: np.ndarray) -> np.ndarray:
    seq_len = len(sequence)
    reverse = reverse_complement(sequence)
    max_codons = seq_len // 3 + 1
    codon_ids = np.full((6, max_codons), 64, dtype=np.int64)

    for frame in range(6):
        current = reverse if frame >= 3 else sequence
        offset = frame % 3
        n_codons = (seq_len - offset) // 3

        for codon_index in range(n_codons):
            position = offset + codon_index * 3
            if position + 2 >= seq_len:
                break

            b1 = int(current[position])
            b2 = int(current[position + 1])
            b3 = int(current[position + 2])

            if b1 < 4 and b2 < 4 and b3 < 4:
                codon_ids[frame, codon_index] = b1 * 16 + b2 * 4 + b3

    return codon_ids


class VirusDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        fragment_length: int,
        cache_dir: str | Path | None = ".cache",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.fragment_length = int(fragment_length)
        self.cache_dir = Path(cache_dir) if cache_dir else None

        sequences, labels = self._load()
        self.sequences = torch.from_numpy(sequences)
        self.labels = torch.from_numpy(labels)

    def _cache_paths(self):
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        stem = self.csv_path.stem
        return (
            self.cache_dir / f"{stem}_{self.fragment_length}bp_sequences.npy",
            self.cache_dir / f"{stem}_{self.fragment_length}bp_labels.npy",
        )

    def _load(self):
        cache_paths = self._cache_paths()
        if cache_paths and all(path.exists() for path in cache_paths):
            return np.load(cache_paths[0]), np.load(cache_paths[1])

        frame = pd.read_csv(self.csv_path)
        sequence_column = "sequence" if "sequence" in frame.columns else "seq"
        label_column = "label" if "label" in frame.columns else "y"

        if sequence_column not in frame.columns or label_column not in frame.columns:
            raise ValueError(
                "CSV must contain sequence and label columns "
                "(seq/y aliases are also accepted)."
            )

        sequences = np.empty((len(frame), self.fragment_length), dtype=np.uint8)
        labels = frame[label_column].to_numpy(dtype=np.int64)

        for index, sequence in enumerate(
            tqdm(frame[sequence_column], desc="Encoding sequences")
        ):
            sequences[index] = encode_sequence(sequence, self.fragment_length)

        if cache_paths:
            try:
                np.save(cache_paths[0], sequences)
                np.save(cache_paths[1], labels)
            except OSError:
                pass

        return sequences, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.sequences[index], int(self.labels[index])


class TrainCollator:
    def __init__(self, reverse_complement_probability: float = 0.5):
        self.reverse_complement_probability = float(reverse_complement_probability)

    def __call__(self, batch):
        nucleotide_ids, codon_ids, labels = [], [], []

        for sequence_tensor, label in batch:
            sequence = sequence_tensor.numpy()
            if random.random() < self.reverse_complement_probability:
                sequence = reverse_complement(sequence)

            nucleotide_ids.append(torch.from_numpy(sequence.astype(np.int64)))
            codon_ids.append(torch.from_numpy(six_frame_codon_ids(sequence)))
            labels.append(label)

        nucleotide_ids = torch.stack(nucleotide_ids)
        nucleotide_mask = nucleotide_ids.eq(4)
        nucleotide_one_hot = (
            F.one_hot(nucleotide_ids, num_classes=5)[..., :4]
            .transpose(1, 2)
            .float()
        )

        codon_ids = torch.stack(codon_ids)
        codon_mask = codon_ids.eq(64)

        return (
            nucleotide_one_hot,
            codon_ids,
            nucleotide_mask,
            codon_mask,
            torch.tensor(labels, dtype=torch.long),
        )


def build_train_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    reverse_complement_probability: float,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=TrainCollator(reverse_complement_probability),
    )
