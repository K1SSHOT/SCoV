from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from tqdm import tqdm

from dataset import VirusDataset, build_train_loader
from losses import SCoVLoss
from model import SCoV
from utils import build_scheduler, mask_nucleotides, set_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train SCoV on fixed-length viral fragments."
    )
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument(
        "--fragment-length",
        required=True,
        type=int,
        choices=(300, 500),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/scov"),
    )

    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--mask-probability", type=float, default=0.05)
    parser.add_argument(
        "--reverse-complement-probability",
        type=float,
        default=0.5,
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument(
        "--contrastive-temperature",
        type=float,
        default=0.1,
    )

    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--non-deterministic", action="store_true")

    return parser.parse_args()


def save_checkpoint(
    path,
    epoch,
    model,
    optimizer,
    scheduler,
    scaler,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
        },
        path,
    )


def main():
    args = parse_args()

    set_seed(
        args.seed,
        deterministic=not args.non_deterministic,
    )

    if not args.train_csv.exists():
        raise FileNotFoundError(
            f"Training CSV not found: {args.train_csv}"
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    dataset = VirusDataset(
        csv_path=args.train_csv,
        fragment_length=args.fragment_length,
        cache_dir=args.cache_dir,
    )

    loader = build_train_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        reverse_complement_probability=(
            args.reverse_complement_probability
        ),
    )

    model = SCoV(
        d_model=args.hidden_dim,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    criterion = SCoVLoss(
        label_smoothing=args.label_smoothing,
        contrastive_weight=args.contrastive_weight,
        temperature=args.contrastive_temperature,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler = build_scheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr_ratio=args.min_lr_ratio,
    )

    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    start_epoch = 0

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(
            checkpoint["model_state_dict"]
        )
        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )
        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )
        scaler.load_state_dict(
            checkpoint["scaler_state_dict"]
        )
        start_epoch = int(
            checkpoint.get("epoch", 0)
        )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Device: {device}")
    print(f"Fragment length: {args.fragment_length} bp")
    print(f"Training fragments: {len(dataset):,}")
    print(f"Trainable parameters: {parameter_count:,}")

    log_path = args.output_dir / "training_log.csv"

    for epoch_index in range(start_epoch, args.epochs):
        epoch = epoch_index + 1
        model.train()

        loss_sum = 0.0
        grad_sum = 0.0
        step_count = 0

        progress = tqdm(
            loader,
            desc=f"Epoch {epoch:03d}/{args.epochs}",
            dynamic_ncols=True,
        )

        for (
            nucleotide_one_hot,
            codon_ids,
            nucleotide_mask,
            codon_mask,
            labels,
        ) in progress:
            nucleotide_one_hot = nucleotide_one_hot.to(
                device,
                non_blocking=True,
            )
            nucleotide_one_hot = mask_nucleotides(
                nucleotide_one_hot,
                args.mask_probability,
            )
            codon_ids = codon_ids.to(
                device,
                non_blocking=True,
            )
            nucleotide_mask = nucleotide_mask.to(
                device,
                non_blocking=True,
            )
            codon_mask = codon_mask.to(
                device,
                non_blocking=True,
            )
            labels = labels.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type="cuda",
                enabled=amp_enabled,
            ):
                logits, fused_features, codon_features = model(
                    nucleotide_one_hot,
                    codon_ids,
                    nucleotide_mask,
                    codon_mask,
                )
                loss = criterion(
                    logits,
                    labels,
                    fused_features,
                    codon_features,
                )

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                args.grad_clip,
            )

            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.detach())
            loss_sum += loss_value
            grad_sum += float(grad_norm)
            step_count += 1

            progress.set_postfix(
                loss=f"{loss_value:.4f}"
            )

        scheduler.step()

        mean_loss = loss_sum / max(step_count, 1)
        mean_grad = grad_sum / max(step_count, 1)

        row = {
            "epoch": epoch,
            "train_loss": mean_loss,
            "grad_norm": mean_grad,
            "learning_rate": scheduler.get_last_lr()[0],
        }

        write_header = not log_path.exists()
        with log_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=row.keys(),
            )
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        if (
            args.save_every > 0
            and epoch % args.save_every == 0
        ):
            save_checkpoint(
                args.output_dir / f"epoch_{epoch:03d}.pt",
                epoch,
                model,
                optimizer,
                scheduler,
                scaler,
            )

    save_checkpoint(
        args.output_dir / "final.pt",
        args.epochs,
        model,
        optimizer,
        scheduler,
        scaler,
    )


if __name__ == "__main__":
    main()
