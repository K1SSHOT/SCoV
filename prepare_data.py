"""Convert explicitly supplied, already split internal CSV files to fixed shards.

No splitting, sampling, sequence trimming or external dataset discovery.
"""
import argparse
import gzip
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import EXPECTED_COUNTS, FRAGMENT_LENGTHS, SPLITS
from utils import sha256_file, write_json

ALIASES = {
    "sequence": ("sequence", "seq"), "label": ("label", "y", "target"),
    "split": ("split", "set", "partition"),
    "length": ("fragment_length", "frag_len", "length"),
}
METADATA_COLUMNS = (
    "fragment_id", "parent_id", "cluster_id", "assembly_accession",
    "sequence_accession", "original_sequence_id", "source", "domain",
    "phylum", "genus", "species", "fragment_start", "fragment_end",
)
LOOKUP = np.full(256, 4, dtype=np.uint8)
for i, c in enumerate("ACGT"):
    LOOKUP[ord(c)] = LOOKUP[ord(c.lower())] = i


def find_column(columns, aliases):
    lower = {str(c).strip().lower(): c for c in columns}
    return next((lower[a] for a in aliases if a in lower), None)


def normalize_labels(series):
    s = series.astype(str).str.strip().str.lower()
    words = {"host": "0", "negative": "0", "nonviral": "0", "non-viral": "0",
             "virus": "1", "viral": "1", "positive": "1"}
    values = pd.to_numeric(s.replace(words), errors="raise").to_numpy()
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError("Labels must be exactly 0 (host) or 1 (virus)")
    return values.astype(np.uint8)


def normalize_split_values(series):
    mapping = {"train": "train", "training": "train", "val": "validation",
               "valid": "validation", "validation": "validation", "dev": "validation",
               "test": "test", "testing": "test"}
    result = series.astype(str).str.strip().str.lower().map(mapping)
    if result.isna().any():
        raise ValueError("Unknown or missing split value")
    return result


def encode_fixed_sequences(series, length):
    if series.isna().any():
        raise ValueError("Missing sequence")
    s = series.astype(str)
    if not s.str.len().eq(length).all():
        raise ValueError(f"Every sequence must have exactly {length} bases")
    if not s.str.fullmatch(r"[ACGTRYSWKMBDHVNacgtryswkmbdhvn]+").all():
        raise ValueError("Sequences must contain ASCII DNA/IUPAC bases only")
    # IUPAC ambiguity maps to 4, matching the original N representation.
    raw = s.to_numpy(dtype=f"S{length}").view(np.uint8).reshape(-1, length)
    return LOOKUP[raw]


def infer_fields(path, length):
    name = path.name.lower()
    match = re.search(r"(?<!\d)(300|500)\s*bp(?!\d)", name)
    inferred_length = int(match.group(1)) if match else None
    match = re.search(r"(?:^|[_\-.])(train|training|validation|val|valid|dev|test|testing)(?:[_\-.]|$)", name)
    inferred_split = normalize_split_values(pd.Series([match.group(1)])).iloc[0] if match else None
    if inferred_length is not None and inferred_length != length:
        raise ValueError(f"Filename length conflicts with --length: {path}")
    return inferred_length, inferred_split


class SplitWriter:
    def __init__(self, root, length, split, shard_size):
        self.root = root / split
        self.root.mkdir()
        self.length, self.split, self.shard_size = length, split, shard_size
        self.seq = np.empty((shard_size, length), dtype=np.uint8)
        self.labels = np.empty(shard_size, dtype=np.uint8)
        self.used = self.total = 0
        self.shards = []
        self.class_counts = np.zeros(2, dtype=np.int64)
        self.metadata = gzip.open(self.root / "metadata.csv.gz", "wt", encoding="utf-8", newline="")
        self.header_written = False

    def add(self, sequences, labels, metadata):
        if len(sequences) != len(labels) or len(labels) != len(metadata):
            raise ValueError("Sequence/label/metadata alignment mismatch")
        metadata = metadata.copy()
        metadata.insert(0, "sample_index", np.arange(self.total + self.used,
                                                   self.total + self.used + len(labels)))
        metadata.to_csv(self.metadata, index=False, header=not self.header_written)
        self.header_written = True
        self.class_counts += np.bincount(labels, minlength=2)
        start = 0
        while start < len(labels):
            n = min(self.shard_size - self.used, len(labels) - start)
            self.seq[self.used:self.used + n] = sequences[start:start + n]
            self.labels[self.used:self.used + n] = labels[start:start + n]
            self.used += n
            start += n
            if self.used == self.shard_size:
                self.flush()

    def flush(self):
        if not self.used:
            return
        number = len(self.shards)
        shard = {"shard": number, "n": self.used,
                 "seq_file": f"seq_{number:05d}.npy", "label_file": f"label_{number:05d}.npy"}
        np.save(self.root / shard["seq_file"], self.seq[:self.used])
        np.save(self.root / shard["label_file"], self.labels[:self.used])
        for key in ("seq_file", "label_file"):
            shard[key + "_sha256"] = sha256_file(self.root / shard[key])
        self.shards.append(shard)
        self.total += self.used
        self.used = 0

    def close(self):
        self.metadata.close()

    def finalize(self, sources):
        self.flush()
        self.close()
        if not self.total:
            raise ValueError(f"Empty {self.split} split")
        manifest = {"format_version": 1, "fragment_length": self.length,
                    "split": self.split, "count": self.total, "shards": self.shards,
                    "class_counts": {"host_0": int(self.class_counts[0]),
                                     "virus_1": int(self.class_counts[1])},
                    "sources": sources, "metadata_file": "metadata.csv.gz",
                    "metadata_sha256": sha256_file(self.root / "metadata.csv.gz")}
        write_json(self.root / "manifest.json", manifest)
        return manifest


def prepare(inputs, cache_dir, length, *, shard_size=250000, chunk_rows=100000,
            strict_expected_counts=True):
    if length not in FRAGMENT_LENGTHS or shard_size <= 0 or chunk_rows <= 0:
        raise ValueError("Invalid length, shard size or chunk size")
    inputs = [Path(p).resolve() for p in inputs]
    if not inputs or len(set(inputs)) != len(inputs):
        raise ValueError("Provide distinct explicit input files")
    for p in inputs:
        if not p.is_file() or not (p.name.endswith(".csv") or p.name.endswith(".csv.gz")):
            raise ValueError(f"Not a CSV/CSV.GZ file: {p}")
    parent = Path(cache_dir).resolve() / "dataset"
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / f"{length}bp"
    if destination.exists():
        raise FileExistsError(f"Cache exists: {destination}; use a new --cache-dir")
    staging = Path(tempfile.mkdtemp(prefix=f".{length}bp-", dir=parent))
    writers = {}
    try:
        writers = {s: SplitWriter(staging, length, s, shard_size) for s in SPLITS}
        sources = []
        for source_index, path in enumerate(inputs):
            inferred_length, inferred_split = infer_fields(path, length)
            before_hash = sha256_file(path)
            row_offset = 0
            for chunk in pd.read_csv(path, chunksize=chunk_rows, low_memory=False):
                columns = {k: find_column(chunk.columns, v) for k, v in ALIASES.items()}
                if not columns["sequence"] or not columns["label"]:
                    raise ValueError(f"Missing sequence/label columns: {path}")
                if columns["length"]:
                    sizes = pd.to_numeric(chunk[columns["length"]], errors="raise").to_numpy()
                    if not np.isfinite(sizes).all() or not (sizes == length).all():
                        raise ValueError(f"All fragment lengths must equal --length {length}: {path}")
                elif inferred_length is None:
                    raise ValueError("Need fragment_length column or a filename containing 300bp/500bp")
                if columns["split"]:
                    splits = normalize_split_values(chunk[columns["split"]])
                    if inferred_split is not None and not splits.eq(inferred_split).all():
                        raise ValueError(f"Filename split conflicts with split column: {path}")
                elif inferred_split is not None:
                    splits = pd.Series(inferred_split, index=chunk.index)
                else:
                    raise ValueError("Need split column or train/validation/test in filename")
                sequences = encode_fixed_sequences(chunk[columns["sequence"]], length)
                labels = normalize_labels(chunk[columns["label"]])
                metadata = chunk.reindex(columns=METADATA_COLUMNS).copy()
                metadata.insert(0, "source_row", np.arange(row_offset, row_offset + len(chunk)))
                metadata.insert(0, "source_index", source_index)
                metadata.insert(0, "source_file", path.name)
                # Missing optional metadata stays empty; source_index/source_row are always available.
                for split in SPLITS:
                    positions = np.flatnonzero(splits.to_numpy() == split)
                    if len(positions):
                        writers[split].add(sequences[positions], labels[positions], metadata.iloc[positions])
                row_offset += len(chunk)
            if sha256_file(path) != before_hash:
                raise RuntimeError(f"Input changed during preprocessing: {path}")
            sources.append({"source_index": source_index, "name": path.name,
                            "sha256": before_hash, "rows": row_offset})
        manifests = {s: writers[s].finalize(sources) for s in SPLITS}
        if strict_expected_counts:
            for s, m in manifests.items():
                expected = EXPECTED_COUNTS[length][s]
                if m["count"] != expected or any(v != expected // 2 for v in m["class_counts"].values()):
                    raise ValueError(f"Paper dataset count/class balance mismatch: {s}: {m['count']}")
        write_json(staging / "dataset_manifest.json", {
            "complete": True, "fragment_length": length,
            "split_fingerprints": {s: sha256_file(staging / s / "manifest.json") for s in SPLITS},
            "sources": sources,
        })
        staging.rename(destination)
        return {s: manifests[s]["count"] for s in SPLITS}
    finally:
        for w in writers.values():
            w.close()
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", required=True, type=Path)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--length", type=int, choices=FRAGMENT_LENGTHS, required=True)
    parser.add_argument("--shard-size", type=int, default=250000)
    parser.add_argument("--chunk-rows", type=int, default=100000)
    parser.add_argument("--allow-count-mismatch", action="store_true",
                        help="For smoke tests/custom data; disables the paper count/balance check")
    args = parser.parse_args()
    print(prepare(args.input, args.cache_dir, args.length, shard_size=args.shard_size,
                  chunk_rows=args.chunk_rows, strict_expected_counts=not args.allow_count_mismatch))

if __name__ == "__main__":
    main()
