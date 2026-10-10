"""Original epoch-wise warmup followed by step-wise cosine decay."""
import math
import torch

def make_scheduler(
    optimizer,
    steps_per_epoch,
    total_epochs,
    warmup_epochs,
    min_ratio,
):
    # Old SCoV warmup semantics:
    # epoch 1 -> 0.5 * base LR; epoch 2 -> 1.0 * base LR; then cosine decay.
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    def lr_lambda(step):
        if warmup_epochs > 0 and step < warmup_steps:
            epoch_idx = min(step // steps_per_epoch, warmup_epochs - 1)
            return (epoch_idx + 1) / warmup_epochs

        denom = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


