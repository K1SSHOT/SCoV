from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from dataset import VirusDataset, six_frame_codon_ids
from model import AblationSCoV


INTERNAL_VAL_RATIO = 0.1
INTERNAL_SEED = 2026


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SCoV on the fixed internal validation split."
    )

    parser.add_argument(
        "--test-csv",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--fragment-length",
        required=True,
        type=int,
        choices=(300, 500),
    )

    parser.add_argument(
        "--weight",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def resolve_weight_path(
    fragment_length: int,
    weight_path: Path | None,
) -> Path:
    if weight_path is not None:
        return weight_path.resolve()

    project_root = Path(__file__).resolve().parent

    filenames = {
        300: "best_Full_SCoV_300bp.pth",
        500: "best_Full_SCoV_500bp.pth",
    }

    return project_root / filenames[fragment_length]


def load_state_dict(
    weight_path: Path,
):
    try:
        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            weight_path,
            map_location="cpu",
        )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "Unsupported checkpoint format."
        )

    state_dict = checkpoint.get(
        "model_state_dict",
        checkpoint,
    )

    clean_state_dict = OrderedDict()

    for key, value in state_dict.items():
        while key.startswith("_orig_mod."):
            key = key[len("_orig_mod."):]

        while key.startswith("module."):
            key = key[len("module."):]

        clean_state_dict[key] = value

    return clean_state_dict


def load_model(
    weight_path: Path,
    device: torch.device,
    hidden_dim: int,
    dropout: float,
):
    if not weight_path.exists():
        raise FileNotFoundError(
            f"Weight file not found: {weight_path}"
        )

    state_dict = load_state_dict(
        weight_path
    )

    use_frame_prior = (
        "cia.gate_strength"
        in state_dict
    )

    model = AblationSCoV(
        d_model=hidden_dim,
        dropout=dropout,
        use_codon=True,
        use_cia=True,
        use_stop_gate=True,
        use_cub_pool=True,
        use_bigru_base=True,
        use_frame_prior=use_frame_prior,
        codon_frame_mode="six",
    ).to(device)

    missing, unexpected = model.load_state_dict(
        state_dict,
        strict=False,
    )

    critical_missing = [
        key
        for key in missing
        if not any(
            key.endswith(suffix)
            for suffix in (
                "running_mean",
                "running_var",
                "num_batches_tracked",
            )
        )
    ]

    if critical_missing:
        raise RuntimeError(
            "Critical model parameters are missing:\n"
            + "\n".join(critical_missing)
        )

    if unexpected:
        raise RuntimeError(
            "Unexpected model parameters were found:\n"
            + "\n".join(unexpected)
        )

    model.eval()

    return model, use_frame_prior


def collate_fn(batch):
    nucleotide_ids = []
    codon_ids = []
    labels = []

    for sequence_tensor, label in batch:
        sequence = sequence_tensor.numpy()

        nucleotide_ids.append(
            torch.from_numpy(
                sequence.astype(
                    np.int64
                )
            )
        )

        codon_ids.append(
            torch.from_numpy(
                six_frame_codon_ids(
                    sequence
                )
            )
        )

        labels.append(
            label
        )

    nucleotide_ids = torch.stack(
        nucleotide_ids
    )

    nucleotide_mask = (
        nucleotide_ids == 4
    )

    nucleotide_one_hot = (
        F.one_hot(
            nucleotide_ids,
            num_classes=5,
        )[..., :4]
        .transpose(
            1,
            2,
        )
        .float()
    )

    codon_ids = torch.stack(
        codon_ids
    )

    codon_mask = (
        codon_ids == 64
    )

    labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    return (
        nucleotide_one_hot,
        codon_ids,
        nucleotide_mask,
        codon_mask,
        labels,
    )


def get_internal_indices(
    n_samples: int,
):
    train_size = int(
        (
            1.0
            - INTERNAL_VAL_RATIO
        )
        * n_samples
    )

    generator = torch.Generator().manual_seed(
        INTERNAL_SEED
    )

    permutation = torch.randperm(
        n_samples,
        generator=generator,
    ).numpy()

    return permutation[
        train_size:
    ].astype(
        np.int64
    )


def build_loader(
    dataset,
    indices,
    batch_size: int,
    num_workers: int,
    device: torch.device,
):
    subset = Subset(
        dataset,
        indices.tolist(),
    )

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )

    return subset, loader


def infer(
    model,
    loader,
    device,
):
    probabilities = []
    labels = []

    use_amp = (
        device.type == "cuda"
    )

    with torch.inference_mode():
        for (
            nucleotide_one_hot,
            codon_ids,
            nucleotide_mask,
            codon_mask,
            batch_labels,
        ) in tqdm(
            loader,
            desc="Inference",
            dynamic_ncols=True,
        ):
            nucleotide_one_hot = nucleotide_one_hot.to(
                device,
                non_blocking=True,
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

            with torch.amp.autocast(
                "cuda",
                enabled=use_amp,
            ):
                logits, _, _ = model(
                    nucleotide_one_hot,
                    codon_ids,
                    nucleotide_mask,
                    codon_mask,
                )

            batch_probabilities = torch.softmax(
                logits.float(),
                dim=1,
            )[:, 1]

            batch_probabilities = torch.nan_to_num(
                batch_probabilities,
                nan=0.5,
            )

            probabilities.extend(
                batch_probabilities
                .cpu()
                .numpy()
                .tolist()
            )

            labels.extend(
                batch_labels
                .numpy()
                .tolist()
            )

    return (
        np.asarray(
            labels,
            dtype=np.int64,
        ),
        np.asarray(
            probabilities,
            dtype=np.float64,
        ),
    )


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
):
    y_pred = (
        y_score > 0.5
    ).astype(
        np.int64
    )

    has_multiple_classes = (
        np.unique(
            y_true
        ).size
        > 1
    )

    metrics = {
        "Accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "Precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "Recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "F1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "MCC": (
            float(
                matthews_corrcoef(
                    y_true,
                    y_pred,
                )
            )
            if has_multiple_classes
            else 0.0
        ),
        "AUROC": (
            float(
                roc_auc_score(
                    y_true,
                    y_score,
                )
            )
            if has_multiple_classes
            else 0.5
        ),
        "AUPRC": (
            float(
                average_precision_score(
                    y_true,
                    y_score,
                )
            )
            if has_multiple_classes
            else 0.5
        ),
    }

    return metrics


def save_predictions(
    output_path: Path,
    indices: np.ndarray,
    y_true: np.ndarray,
    y_score: np.ndarray,
):
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    y_pred = (
        y_score > 0.5
    ).astype(
        np.int64
    )

    frame = pd.DataFrame(
        {
            "sample_id": indices,
            "y_true": y_true,
            "y_score": y_score,
            "y_pred": y_pred,
        }
    )

    frame.to_csv(
        output_path,
        index=False,
    )


def set_deterministic():
    np.random.seed(
        INTERNAL_SEED
    )

    torch.manual_seed(
        INTERNAL_SEED
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            INTERNAL_SEED
        )

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(
    requested_device: str | None,
):
    if requested_device is not None:
        return torch.device(
            requested_device
        )

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def main():
    args = parse_args()

    if not args.test_csv.exists():
        raise FileNotFoundError(
            f"CSV file not found: {args.test_csv}"
        )

    set_deterministic()

    device = resolve_device(
        args.device
    )

    weight_path = resolve_weight_path(
        args.fragment_length,
        args.weight,
    )

    dataset = VirusDataset(
        csv_path=args.test_csv,
        fragment_length=args.fragment_length,
        cache_dir=args.cache_dir,
    )

    internal_indices = get_internal_indices(
        len(dataset)
    )

    internal_dataset, loader = build_loader(
        dataset=dataset,
        indices=internal_indices,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    model, use_frame_prior = load_model(
        weight_path=weight_path,
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    y_true, y_score = infer(
        model=model,
        loader=loader,
        device=device,
    )

    metrics = compute_metrics(
        y_true=y_true,
        y_score=y_score,
    )

    if args.output is None:
        output_path = (
            Path(__file__).resolve().parent
            / "outputs"
            / f"internal_predictions_{args.fragment_length}bp.csv"
        )
    else:
        output_path = args.output

    save_predictions(
        output_path=output_path,
        indices=internal_indices,
        y_true=y_true,
        y_score=y_score,
    )

    print()
    print(f"Device: {device}")
    print(f"CSV: {args.test_csv}")
    print(f"Weight: {weight_path}")
    print(f"Fragment length: {args.fragment_length} bp")
    print(f"Frame prior: {use_frame_prior}")
    print(f"Full dataset size: {len(dataset):,}")
    print(f"Internal validation size: {len(internal_dataset):,}")
    print(f"Internal seed: {INTERNAL_SEED}")
    print(f"Output: {output_path}")
    print()

    for name, value in metrics.items():
        print(
            f"{name}: {value:.6f}"
        )


if __name__ == "__main__":
    main()
