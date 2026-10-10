"""Train SCoV and select the checkpoint using internal validation AUROC."""
import argparse
from pathlib import Path

import csv
import math
import os
import platform

import numpy as np
import torch
from tqdm import tqdm

from config import EXPECTED_COUNTS, FRAGMENT_LENGTHS, load_config
from data import DeterministicBlockShuffleSampler, ShardedSequenceDataset
from encoding import tokens_to_model_inputs
from engine import (build_loader, build_model, cache_fingerprints, capture_rng_state,
                    evaluate, resolve_device, restore_rng_state, set_global_seed)
from model import HexaCrossLoss, MODEL_VERSION
from schedule import make_scheduler
from utils import save_checkpoint, sha256_file, write_json


def train_epoch(model, loader, optimizer, scheduler, scaler, criterion, cfg, device):
    model.train()
    total_loss, seen = 0.0, 0
    for x_nt, labels in tqdm(loader, desc="train", leave=False):
        labels = labels.to(device).long()
        inputs = tokens_to_model_inputs(x_nt.to(device), needs_codon=True, training=True, cfg=cfg)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=scaler.is_enabled()):
            logits, fused, codon = model(*inputs)
            loss = criterion(logits, labels, fused, codon)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip,
                                      error_if_nonfinite=not scaler.is_enabled())
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # AMP overflow can skip the optimizer update; do not advance LR in that case.
        if scaler.get_scale() >= previous_scale:
            scheduler.step()
        n = len(labels)
        total_loss += float(loss.detach()) * n
        seen += n
    if seen != len(loader.dataset):
        raise RuntimeError("Training did not cover every sample exactly once")
    return total_loss / seen


def source_hashes():
    root = Path(__file__).parent
    return {p.name: sha256_file(p) for p in sorted(root.glob("*.py"))}


def save_history(path, rows):
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def export_best(run_dir, checkpoint):
    save_checkpoint(run_dir / "best.pt", {
        "model_version": MODEL_VERSION, "run_config": checkpoint["run_config"],
        "epoch": checkpoint["best_epoch"], "model_state_dict": checkpoint["best_state"],
        "val_metrics": checkpoint["best_metrics"],
    })
    write_json(run_dir / "validation_best_metrics.json", {
        "best_epoch": checkpoint["best_epoch"], "metrics": checkpoint["best_metrics"]})


def train_run(cfg, cache_dir, output_dir, length, seed, device_name="auto", *,
              resume=False, verify_cache=False, stop_after_epoch=None):
    device = resolve_device(device_name)
    set_global_seed(seed, cfg.strict_determinism)
    run_dir = Path(output_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not resume:
        raise FileExistsError("Output directory is not empty; use --resume or a new --output-dir")
    run_dir.mkdir(parents=True, exist_ok=True)
    fingerprints = cache_fingerprints(cache_dir, length)
    datasets = {s: ShardedSequenceDataset(cache_dir, length, s, cfg.max_open_shards, verify_cache)
                for s in ("train", "validation")}
    for s, ds in datasets.items():
        if cfg.strict_expected_counts:
            expected = EXPECTED_COUNTS[length][s]
            if len(ds) != expected or any(v != expected // 2 for v in ds.manifest["class_counts"].values()):
                raise ValueError(f"Paper dataset count/class balance mismatch: {s}")
        if any(v == 0 for v in ds.manifest["class_counts"].values()):
            raise ValueError(f"Training and validation require both classes: {s}")
    run_config = {"config": cfg.to_dict(), "fragment_length": length, "seed": seed,
                  "model_version": MODEL_VERSION, "split_fingerprints": fingerprints,
                  "source_hashes": source_hashes(), "selection_metric": "AUROC",
                  "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                              "numpy": np.__version__, "cuda": torch.version.cuda,
                              "device": str(device), "platform": platform.platform(),
                              "num_threads": torch.get_num_threads()}}
    model = build_model(cfg, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sampler = DeterministicBlockShuffleSampler(datasets["train"], seed, cfg.block_shuffle_size)
    steps = math.ceil(len(datasets["train"]) / cfg.batch_size)
    scheduler = make_scheduler(optimizer, steps, cfg.epochs, cfg.warmup_epochs, cfg.min_lr_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.use_amp and device.type == "cuda")
    criterion = HexaCrossLoss(cfg.cls_weight, cfg.contra_weight, cfg.label_smoothing, cfg.supcon_temperature)
    start_epoch, best_value, best_epoch = 0, -float("inf"), 0
    best_state, best_metrics, history = None, None, []
    if resume:
        # Resume files contain Python/NumPy RNG state. Load only your own trusted files.
        checkpoint = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
        if checkpoint["run_config"] != run_config:
            raise ValueError("Resume configuration, dataset, source code or runtime differs")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch, best_epoch = checkpoint["epoch"], checkpoint["best_epoch"]
        best_value, best_state = checkpoint["best_value"], checkpoint["best_state"]
        best_metrics, history = checkpoint["best_metrics"], checkpoint["history"]
        restore_rng_state(checkpoint["rng_state"])
        # last.pt is authoritative if interruption happened during export/history writing.
        export_best(run_dir, checkpoint)
        save_history(run_dir / "history.csv", history)
    write_json(run_dir / "config.json", run_config)
    end_epoch = cfg.epochs if stop_after_epoch is None else min(cfg.epochs, stop_after_epoch)
    for epoch in range(start_epoch, end_epoch):
        sampler.set_epoch(epoch)
        # Epoch-derived loader seeds keep resume independent of worker lifecycle.
        loader = build_loader(datasets["train"], cfg, train=True, sampler=sampler, seed=seed + epoch)
        loss = train_epoch(model, loader, optimizer, scheduler, scaler, criterion, cfg, device)
        val, _, _ = evaluate(model, datasets["validation"], cfg, device, seed=epoch)
        value = val["AUROC"]
        if value is None or not math.isfinite(value):
            raise ValueError("Validation AUROC is undefined")
        improved = value > best_value
        if improved:
            best_value, best_epoch, best_metrics = value, epoch + 1, val
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch + 1, "train_loss": loss,
                        "lr": scheduler.get_last_lr()[0], **{f"val_{k}": v for k, v in val.items()},
                        "is_best": int(improved)})
        checkpoint = {"epoch": epoch + 1, "run_config": run_config,
                      "model_state_dict": model.state_dict(),
                      "optimizer_state_dict": optimizer.state_dict(),
                      "scheduler_state_dict": scheduler.state_dict(),
                      "scaler_state_dict": scaler.state_dict(), "rng_state": capture_rng_state(),
                      "best_value": best_value, "best_epoch": best_epoch,
                      "best_state": best_state, "best_metrics": best_metrics, "history": history}
        save_checkpoint(run_dir / "last.pt", checkpoint)
        export_best(run_dir, checkpoint)
        save_history(run_dir / "history.csv", history)
        print(f"Epoch {epoch+1}/{cfg.epochs}: loss={loss:.6f}, val_AUROC={value:.6f}, best_epoch={best_epoch}")
    if max(start_epoch, end_epoch) == cfg.epochs:
        write_json(run_dir / "TRAINING_COMPLETE.json", {"best_epoch": best_epoch, "best_AUROC": best_value})
    return run_dir / "best.pt"

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", type=int, choices=FRAGMENT_LENGTHS, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=Path, default=Path(__file__).parent / "configs" / "paper.json")
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    parser.add_argument("--resume", action="store_true", help="Restore the last completed epoch")
    parser.add_argument("--verify-cache", action="store_true", help="Recompute shard checksums before training")
    parser.add_argument("--stop-after-epoch", type=int, help="Optional planned stop; use --resume to continue")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= cfg.epochs:
        parser.error("--stop-after-epoch must be between 1 and configured epochs")
    best = train_run(cfg, args.cache_dir, args.output_dir, args.length, args.seed,
                     args.device, resume=args.resume, verify_cache=args.verify_cache,
                     stop_after_epoch=args.stop_after_epoch)
    print(f"Best checkpoint: {best}")


if __name__ == "__main__":
    main()
