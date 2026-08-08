# Data

Training data are expected as fixed-length CSV files.

Required columns:

```text
sequence,label
ACGT...,1
TGCA...,0
```

`1` denotes virus and `0` denotes host.

The 300 bp and 500 bp datasets use the same training code; select the length with
`--fragment-length`.

The full datasets are not included here because of their size.
