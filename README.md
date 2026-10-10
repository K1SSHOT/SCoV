# SCoV

Code for training SCoV and evaluating it on the fixed internal test set.

## Contents

- [SCoV](#scov)
  - [Contents](#contents)
  - [1. Overview](#1-overview)
  - [2. Setup Environment](#2-setup-environment)
  - [3. Quick Start](#3-quick-start)
    - [Prepare the Data](#prepare-the-data)
    - [Train the Model](#train-the-model)
    - [Evaluate the Internal Test Set](#evaluate-the-internal-test-set)
  - [4. Output Explanation](#4-output-explanation)
    - [Training Outputs](#training-outputs)
    - [Internal Test Outputs](#internal-test-outputs)
  - [5. Citation](#5-citation)
  - [6. Contact](#6-contact)

## 1. Overview

SCoV is a deep learning model for identifying viral DNA fragments. It combines nucleotide features and six-frame codon representations using a stop-codon-derived frame prior and cross-modal attention.

This repository supports training and internal test evaluation for fixed-length fragments of **300 bp** and **500 bp**. Labels are **0 for host / nonviral** and **1 for virus / viral**. The training split is used for parameter updates, the validation split selects the best model by AUROC, and the internal test split is evaluated separately after training.

## 2. Setup Environment

The CPU workflow has been checked on Linux with Python 3.12 and PyTorch 2.5.1+cpu.

After downloading and extracting the code, enter the project directory and create an environment:

```bash
cd scov_repro
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the CPU version of PyTorch and the remaining dependencies.
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For GPU training, install a compatible CUDA version of PyTorch 2.5.1 using the [official instructions](https://pytorch.org/get-started/previous-versions/#v251), then install the remaining dependencies. `--device auto` uses CUDA when available and CPU otherwise. GPU/AMP and multi-worker execution have not been verified in the validation environment.

## 3. Quick Start

Run the following commands from the project directory. Replace the example data paths with your fixed internal dataset files.

### Prepare the Data

Input files must be CSV or CSV.GZ files containing already assigned train, validation, and test splits.

| Column | Description |
| --- | --- |
| `sequence` | DNA sequence with exactly the selected fragment length; ASCII DNA/IUPAC bases are accepted |
| `label` | Binary label: 0 = host / nonviral, 1 = virus / viral |
| `split` | `train`, `validation`, or `test` |
| `fragment_length` | 300 or 500; must match `--length` |
| `fragment_id` | Optional identifier for tracing predictions back to the input |

When `split` and `fragment_length` are omitted, they can be inferred from filenames such as `train_300bp.csv`, `validation_300bp.csv`, and `test_300bp.csv`.

```bash
python prepare_data.py --length 300 --input data/train_300bp.csv data/validation_300bp.csv data/test_300bp.csv --cache-dir cache
```

Preprocessing preserves the existing splits and converts the sequences to NumPy shards. It does not resample, trim, or pad sequences. The default protocol checks the fixed sample counts defined in `config.py` and equal host/virus counts within each split. Invalid inputs cause an error.

Prepare each fragment length separately. Both lengths can share the same cache root. Input files must be specified explicitly; existing caches are not overwritten.

### Train the Model

```bash
python train.py --length 300 --seed 42 --config configs/paper.json --cache-dir cache --output-dir outputs/300bp/seed_42 --device auto
```

**Training options**

| Option | Description |
| --- | --- |
| `--length` | Fragment length: 300 or 500; required |
| `--seed` | Random seed; default: 42 |
| `--config` | JSON experiment configuration; default: `configs/paper.json` |
| `--cache-dir` | Prepared dataset cache; default: `cache` |
| `--output-dir` | Directory for checkpoints and training records; required |
| `--device` | `auto`, `cpu`, `cuda`, or a CUDA device such as `cuda:0`; default: `auto` |
| `--resume` | Resume from `last.pt` at a completed epoch boundary |
| `--verify-cache` | Recompute shard and metadata checksums before training |

The main configuration uses AdamW, 20 epochs, batch size 512, learning rate `2e-4`, weight decay `1e-4`, and two warmup epochs followed by cosine decay. Model and loss settings are stored in `configs/paper.json`. No YAML file is required.

For 500 bp experiments, prepare the corresponding inputs, use `--length 500`, and choose a separate output directory. The configured experiment seeds are `42`, `123`, `2024`, `3407`, and `7777`; run each seed separately.

To continue an interrupted run, repeat the same training command with `--resume`. Keep the configuration, Python source files, dataset manifests, and recorded runtime environment unchanged.

### Evaluate the Internal Test Set

```bash
python infer.py --checkpoint outputs/300bp/seed_42/best.pt --cache-dir cache --output-dir outputs/300bp/seed_42/internal_test --device auto
```

**Inference options**

| Option | Description |
| --- | --- |
| `--checkpoint` | Path to the `best.pt` checkpoint produced by `train.py`; required |
| `--cache-dir` | Cache containing the same fixed internal test split recorded during training; default: `cache` |
| `--output-dir` | New directory for test predictions and metrics; required |
| `--device` | Inference device; default: `auto` |
| `--batch-size` | Optional override of the evaluation batch size stored in the checkpoint |
| `--num-workers` | Optional override of the data-loader worker count stored in the checkpoint |
| `--verify-cache` | Recompute test-shard and metadata checksums before inference |

Inference loads the validation-selected model and evaluates the complete internal test split without augmentation or shuffling. The viral score is the softmax probability of class 1; with the default configuration, scores greater than or equal to **0.5** are classified as viral. The threshold is stored in the training configuration and is not tuned on the test set.

Use checkpoints produced by the accompanying training code. Older checkpoints without the required configuration and dataset metadata need a separately validated conversion. Existing inference output directories are not overwritten.

## 4. Output Explanation

### Training Outputs

| File | Description |
| --- | --- |
| `best.pt` | Model selected by the highest validation AUROC; ties retain the earlier epoch |
| `last.pt` | Model, optimizer, scheduler, AMP scaler, random states, and history for resuming training |
| `config.json` | Effective configuration, dataset fingerprints, source hashes, and runtime metadata |
| `history.csv` | Per-epoch training loss, learning rate, and validation metrics |
| `validation_best_metrics.json` | Best validation epoch and its metrics |
| `TRAINING_COMPLETE.json` | Training completion marker |

### Internal Test Outputs

**`test_predictions.npz`** contains one prediction for each test fragment, in cache order:

| Array | Description |
| --- | --- |
| `sample_index` | Row index within the current internal test split |
| `y_true` | Ground-truth label: 0 or 1 |
| `y_prob` | Predicted viral probability, between 0 and 1 |
| `y_pred` | Predicted label obtained with the saved decision threshold |

Use `sample_index` to join predictions to the corresponding test `metadata.csv.gz`, which records available fragment identifiers and source rows.

**`test_metrics.json`** reports Accuracy, Precision, Recall, F1, AUROC, AUPRC, and MCC, along with experiment identifiers. AUPRC uses average precision. AUROC is `null` for a single-class test set, and AUPRC is `null` when no positive samples are present.

**`inference_config.json`** records the checkpoint hash, test fingerprint, inference settings, and metadata path.

Reproducing paper results requires the exact internal data, fixed splits, experiment settings, and software environment. The supplied dependency versions describe the code-checking environment; they do not establish the original paper environment or demonstrate reproduction of its reported metrics.

## 5. Citation

Please cite the SCoV paper when using this code in your research.

*The paper citation and BibTeX entry will be added by the repository authors.*

## 6. Contact

If you have any questions, please post an issue or email us: tangwentao@zhku.edu.cn