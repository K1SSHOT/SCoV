"""Evaluate a saved SCoV checkpoint on its fixed internal test split only."""
import argparse
import json
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from config import Config, EXPECTED_COUNTS
from data import ShardedSequenceDataset
from engine import build_model, cache_fingerprints, evaluate, resolve_device, set_global_seed
from model import MODEL_VERSION
from utils import sha256_file, write_json


def infer(checkpoint_path, cache_dir, output_dir, device_name="auto", *,
          batch_size=None, num_workers=None, verify_cache=False):
    device = resolve_device(device_name)
    # best.pt contains tensor weights and primitive metadata only.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["model_version"] != MODEL_VERSION:
        raise ValueError("Checkpoint model version mismatch")
    run = checkpoint["run_config"]
    for filename in ("model.py", "encoding.py"):
        if sha256_file(Path(__file__).parent / filename) != run["source_hashes"][filename]:
            raise ValueError(f"Inference source differs from training: {filename}")
    cfg = Config(**run["config"])
    original_cfg = cfg
    overrides = {}
    if batch_size is not None:
        overrides["eval_batch_size"] = batch_size
    if num_workers is not None:
        overrides["num_workers"] = num_workers
    cfg = replace(cfg, **overrides)
    length = run["fragment_length"]
    fingerprints = cache_fingerprints(cache_dir, length)
    if fingerprints["test"] != run["split_fingerprints"]["test"]:
        raise ValueError("Test split differs from the training experiment")
    set_global_seed(run["seed"], cfg.strict_determinism)
    dataset = ShardedSequenceDataset(cache_dir, length, "test", cfg.max_open_shards, verify_cache)
    if cfg.strict_expected_counts:
        expected = EXPECTED_COUNTS[length]["test"]
        if len(dataset) != expected or any(v != expected // 2 for v in dataset.manifest["class_counts"].values()):
            raise ValueError("Paper internal test count/class balance mismatch")
    model = build_model(cfg, device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}; choose a new --output-dir")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish outputs as a complete directory only after successful inference.
    with tempfile.TemporaryDirectory(prefix=".inference-", dir=destination.parent) as tmp:
        staging = Path(tmp) / "results"
        staging.mkdir()
        metrics, y_true, y_prob = evaluate(model, dataset, cfg, device)
        np.savez(staging / "test_predictions.npz", sample_index=np.arange(len(dataset), dtype=np.int64),
                 y_true=y_true, y_prob=y_prob, y_pred=(y_prob >= cfg.threshold).astype(np.uint8))
        write_json(staging / "test_metrics.json", {
            **metrics, "best_epoch": checkpoint["epoch"], "seed": run["seed"],
            "fragment_length": length, "model": "SCoV"})
        write_json(staging / "inference_config.json", {
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "test_fingerprint": dataset.fingerprint, "device": str(device),
            "training_config": original_cfg.to_dict(), "inference_config": cfg.to_dict(),
            "metadata_file": str(dataset.root.resolve() / "metadata.csv.gz"),
            "join_key": "sample_index", "torch": str(torch.__version__)})
        staging.rename(destination)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--verify-cache", action="store_true")
    args = parser.parse_args()
    metrics = infer(args.checkpoint, args.cache_dir, args.output_dir, args.device,
                    batch_size=args.batch_size, num_workers=args.num_workers,
                    verify_cache=args.verify_cache)
    print(json.dumps(metrics, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
