"""Shared train/evaluation operations. Inference does not import train.py."""
import json
import os
import random
from pathlib import Path

# Set before CUDA context creation; strict mode fails on unsupported operations.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import seed_worker
from encoding import tokens_to_model_inputs
from metrics import compute_metrics
from model import SCoV
from utils import sha256_file


def set_global_seed(seed, strict=True):
    if not 0 <= seed < 2**32:
        raise ValueError("seed must be in [0, 2**32)")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=not strict)


def capture_rng_state():
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch_cpu": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Only cpu/cuda devices are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def cache_fingerprints(cache_dir, length):
    root = Path(cache_dir) / "dataset" / f"{length}bp"
    path = root / "dataset_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing complete cache: {path}; run prepare_data.py")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest["fragment_length"] != length:
        raise ValueError("Incomplete cache or fragment-length mismatch")
    fingerprints = {s: sha256_file(root / s / "manifest.json") for s in ("train", "validation", "test")}
    if fingerprints != manifest["split_fingerprints"]:
        raise ValueError("Split manifest checksum mismatch; rebuild cache")
    return fingerprints


def build_model(cfg, device):
    return SCoV(d_model=cfg.hidden_dim, num_heads=cfg.num_heads, dropout=cfg.dropout).to(device)


def build_loader(dataset, cfg, *, train=False, sampler=None, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=cfg.batch_size if train else cfg.eval_batch_size,
                      sampler=sampler, shuffle=False, num_workers=cfg.num_workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=seed_worker,
                      generator=generator, persistent_workers=False, drop_last=False,
                      multiprocessing_context="spawn" if cfg.num_workers > 0 else None)


@torch.inference_mode()
def evaluate(model, dataset, cfg, device, *, seed=0, loader=None):
    model.eval()
    loader = loader if loader is not None else build_loader(dataset, cfg, seed=seed)
    y_true = np.empty(len(dataset), dtype=np.uint8)
    y_prob = np.empty(len(dataset), dtype=np.float32)
    offset = 0
    for x_nt, labels in tqdm(loader, desc=dataset.split, leave=False):
        inputs = tokens_to_model_inputs(x_nt.to(device), needs_codon=True, training=False, cfg=cfg)
        with torch.autocast(device_type=device.type, enabled=cfg.use_amp and device.type == "cuda"):
            logits, _, _ = model(*inputs)
        prob = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
        n = len(prob)
        y_true[offset:offset + n] = labels.numpy()
        y_prob[offset:offset + n] = prob
        offset += n
    if offset != len(dataset):
        raise RuntimeError("Evaluation did not cover every sample exactly once")
    return compute_metrics(y_true, y_prob, cfg.threshold), y_true, y_prob


