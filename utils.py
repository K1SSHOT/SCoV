from __future__ import annotations

import math
import os
import random

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def set_seed(seed: int, deterministic: bool = True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def mask_nucleotides(nucleotide_one_hot: torch.Tensor, probability: float):
    if probability <= 0:
        return nucleotide_one_hot

    mask = (
        torch.rand(
            nucleotide_one_hot.size(0),
            1,
            nucleotide_one_hot.size(2),
            device=nucleotide_one_hot.device,
        )
        < probability
    )
    return nucleotide_one_hot.masked_fill(mask, 0.0)


def build_scheduler(
    optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float,
):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)

        progress = (
            epoch - warmup_epochs
        ) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda)
