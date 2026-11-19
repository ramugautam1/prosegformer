# Commands

All commands run from the repository root after `pip install -e ".[dev]"`.

## Setup and tests

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ".[nifti]" to read NIfTI files
pytest -q                          # unit tests, a tiny training run and the command-line tools, about 15 s on a CPU
pytest -q -m "not slow"            # unit tests only
```

## Segmentation (inference)

A movie with a channel axis, with the default model:

```bash
prosegformer-segment --image movie.tif --channel 0 --out results/movie
```

A single stack, ProSegFormer-Lite, objects for ProTrackFormer (6-connected, at least 6 voxels):

```bash
prosegformer-segment --image stack.tif --checkpoint checkpoints/prosegformer_lite.pt \
    --connectivity 6 --min-size 6 --out results/stack
```

Options:

- `--frames 0:39` segments time points 0 to 39 only
- `--threshold 0.5` replaces the threshold chosen on validation
- `--output crop` reads the output trained on the tracking crops (instance masks drawn one punctum at a time) instead of
  the default whole-field output
- `--no-mirror` skips mirror averaging (four times faster, slightly less accurate)
- `--voxel-size 0.5 0.124 0.124` sets the voxel size (z, y, x in µm) when the file has none
- `--device cpu` forces the CPU

## Evaluation

The released model on the Bazooka test sets, with 95% bootstrap intervals:

```bash
prosegformer-evaluate --checkpoint checkpoints/prosegformer.pt --data data/bazooka --split test
```

Zero-shot on the other proteins (every stack, frame output):

```bash
prosegformer-evaluate --checkpoint checkpoints/prosegformer.pt --data data/gish --split all
prosegformer-evaluate --checkpoint checkpoints/prosegformer.pt --data data/ecad --split all
```

Add `--json scores.json` to keep the scores, or `--split val` to score the validation data.

## Training

ProSegFormer on Bazooka, from the pretrained encoder (the released checkpoint is seed 12):

```bash
prosegformer-train configs/prosegformer.yaml --seed 12 --out runs/prosegformer_s12
```

ProSegFormer-Lite:

```bash
prosegformer-train configs/prosegformer_lite.yaml --seed 12 --out runs/prosegformer_lite_s12
```

Each run writes `model.pt` (the checkpoint used by `prosegformer-segment` and `prosegformer-evaluate`, with the
thresholds chosen on validation), `log.jsonl` (validation every 1,000 steps) and `result.json`. Training takes about
1.5 hours on one GPU (a Titan RTX); retrained from the same seed, the released model's test scores are reproduced
within about 0.01.

Three seeds, as in the paper:

```bash
for s in 10 11 12; do
    prosegformer-train configs/prosegformer.yaml --seed $s --out runs/prosegformer_s$s
    prosegformer-evaluate --checkpoint runs/prosegformer_s$s/model.pt --data data/bazooka --split test \
        --json runs/prosegformer_s$s/test.json
done
```

Override any setting from the command line:

```bash
prosegformer-train configs/prosegformer.yaml --out runs/test --set optim.steps=500 eval.every=250
```

Without pretraining:

```bash
prosegformer-train configs/prosegformer.yaml --out runs/prosegformer_scratch --set init=null
```

## Fine-tuning on another protein

Fine-tune the Bazooka model on the Gish stacks, one cross-validation fold at a time, and score the held-out stacks:

```bash
for k in 0 1 2; do
    prosegformer-train configs/finetune_gish.yaml --fold $k --out runs/gish_f$k
    prosegformer-evaluate --checkpoint runs/gish_f$k/model.pt --data data/gish --split test --fold $k
done
```

`configs/finetune_ecad.yaml` does the same for E-cadherin. For your own protein, put its annotated stacks in a dataset
folder with the same layout (see [data/README.md](data/README.md)) and point `data:` at it.

## Pretraining

Masked-image pretraining on a folder of unannotated T x Z x Y x X movies (the released encoders used six Bazooka
movies, 1,664 frames, which are not part of the repository):

```bash
prosegformer-pretrain configs/pretrain.yaml --movies movies/ --out runs/pretrain
prosegformer-train configs/prosegformer.yaml --out runs/prosegformer --set init=runs/pretrain/pretrained.pt
```

Pretraining runs 30,000 steps, about four hours on one GPU; `configs/pretrain_lite.yaml` pretrains ProSegFormer-Lite.

## Python

```python
from prosegformer.data import load_split
from prosegformer.inference import Segmenter, load_checkpoint
from prosegformer.io import read_stack

seg = Segmenter("checkpoints/prosegformer.pt")
probability, labels = seg(read_stack("stack.tif")[0])

model, info = load_checkpoint("checkpoints/prosegformer.pt")   # the network and its thresholds
volumes = load_split("data/bazooka", "test")                     # normalized volumes with their labels
```
