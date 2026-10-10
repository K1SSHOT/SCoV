"""Validated, portable experiment configuration; no machine-specific paths."""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

FRAGMENT_LENGTHS = (300, 500)
SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = {
    300: {"train": 12198958, "validation": 1508790, "test": 1576396},
    500: {"train": 7302162, "validation": 903048, "test": 943676},
}
PAPER_SEEDS = (42, 123, 2024, 3407, 7777)

@dataclass(frozen=True)
class Config:
    epochs: int = 20
    batch_size: int = 512
    eval_batch_size: int = 512
    num_workers: int = 4
    hidden_dim: int = 128
    num_heads: int = 4
    dropout: float = 0.2
    lr: float = 2e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    min_lr_ratio: float = 0.01
    grad_clip: float = 1.0
    mask_prob: float = 0.05
    rc_prob: float = 0.5
    label_smoothing: float = 0.1
    supcon_temperature: float = 0.1
    cls_weight: float = 1.0
    contra_weight: float = 0.1
    threshold: float = 0.5
    use_amp: bool = True
    strict_determinism: bool = True
    strict_expected_counts: bool = True
    block_shuffle_size: int = 64
    max_open_shards: int = 64

    def __post_init__(self):
        import math
        integer_names = {"epochs", "batch_size", "eval_batch_size", "num_workers",
                         "hidden_dim", "num_heads", "warmup_epochs",
                         "block_shuffle_size", "max_open_shards"}
        boolean_names = {"use_amp", "strict_determinism", "strict_expected_counts"}
        for f in fields(self):
            v = getattr(self, f.name)
            if f.name in boolean_names:
                if type(v) is not bool:
                    raise ValueError(f"{f.name} must be a boolean")
            elif f.name in integer_names:
                minimum = 0 if f.name in {"num_workers", "warmup_epochs"} else 1
                if type(v) is not int or v < minimum:
                    raise ValueError(f"{f.name} must be an integer >= {minimum}")
            elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                raise ValueError(f"{f.name} must be a finite number")
        if self.hidden_dim % 2 or self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be even and divisible by num_heads")
        if self.warmup_epochs > self.epochs:
            raise ValueError("warmup_epochs cannot exceed epochs")
        for name in ("dropout", "label_smoothing"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("mask_prob", "rc_prob", "threshold", "min_lr_ratio"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        for name in ("lr", "grad_clip", "supcon_temperature", "cls_weight"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.weight_decay < 0 or self.contra_weight < 0:
            raise ValueError("weight_decay and contra_weight must be nonnegative")

    def to_dict(self):
        return asdict(self)


def load_config(path=None):
    values = {} if path is None else json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object")
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
    return Config(**values)
