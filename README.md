# SCoV

Training code for **SCoV: a frame-gated cross-modal model for identifying viral sequences in short metagenomic fragments**.

## Repository structure

```text
SCoV/
├── data/
│   └── README.md
├── train.py
├── model.py
├── dataset.py
├── losses.py
├── utils.py
├── scov.yaml
├── LICENSE
├── .gitignore
└── README.md
```

## Environment

```bash
conda env create -f scov.yaml
conda activate scov
```

## Training

The same script is used for both fragment lengths.

500 bp:

```bash
python train.py \
  --train-csv /path/to/500bp.csv \
  --fragment-length 500 \
  --output-dir outputs/scov_500bp
```

300 bp:

```bash
python train.py \
  --train-csv /path/to/300bp.csv \
  --fragment-length 300 \
  --output-dir outputs/scov_300bp
```

The training CSV must contain `sequence` and `label` columns. `label=1` denotes virus and `label=0` denotes host.

Run `python train.py --help` for optional training arguments.

## License

MIT License.
