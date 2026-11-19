# ProSegFormer

Segmentation of protein puncta in 3D live microscopy.

Protein puncta, such as Par-3/Bazooka clusters at the junctions of *Drosophila* epithelial cells, are a few hundred
nanometers across: four to six pixels wide and one to three optical sections thick. Segmenters that turn the image into
tokens of that size, or predict at reduced resolution, find the puncta but cannot draw them. ProSegFormer keeps the
resolution of the data:

- self-attention starts at full in-plane resolution, in windows that span every slice
- the depth axis is never downsampled, so puncta in neighboring sections stay apart
- masked-image pretraining on unannotated movies of the same protein
- one output per annotation convention, so whole-field binary masks and instance masks drawn one punctum at a time can
  be learned together without averaging their styles
- ProSegFormer-Lite: the same network without self-attention, equally accurate and 15% smaller

The repository also contains the segmentation ground truth for three proteins (Bazooka, Gish and E-cadherin).

## Results

Bazooka test sets: one held-out frame (392 puncta) and 8 held-out tracking crops from another embryo (320 crop-frames,
593 puncta). Instance F1 matches objects one-to-one at an IoU threshold; detection F1 matches centroids within
0.5 µm. Mean ± s.d. over three training seeds, with mirror averaging.

| model | params (M) | frame F1@0.5 | frame F1@0.7 | frame det. F1 | crop F1@0.5 | crop F1@0.7 | crop det. F1 |
|---|---|---|---|---|---|---|---|
| nnU-Net v2 ResEnc-M | -- | 0.82 | 0.71 | 0.87 | 0.82 | 0.74 | 0.83 |
| 3D U-Net (same recipe) | 3.1 | 0.87 ± 0.00 | 0.74 ± 0.01 | 0.88 ± 0.01 | 0.85 ± 0.01 | 0.77 ± 0.01 | 0.86 ± 0.01 |
| SwinUNETR (same recipe) | 18.3 | 0.87 ± 0.01 | 0.72 ± 0.02 | 0.89 ± 0.01 | 0.85 ± 0.01 | 0.76 ± 0.02 | 0.87 ± 0.00 |
| SegFormer3D, z kept (same recipe) | 3.5 | 0.82 ± 0.01 | 0.66 ± 0.02 | 0.86 ± 0.00 | 0.84 ± 0.00 | 0.69 ± 0.03 | 0.87 ± 0.01 |
| **ProSegFormer** | 1.2 | **0.89 ± 0.01** | **0.82 ± 0.01** | **0.91 ± 0.01** | 0.87 ± 0.01 | **0.81 ± 0.01** | 0.88 ± 0.01 |
| **ProSegFormer-Lite** | 1.0 | 0.88 ± 0.01 | 0.80 ± 0.01 | 0.90 ± 0.01 | **0.88 ± 0.00** | **0.81 ± 0.00** | 0.88 ± 0.01 |

SegFormer3D with z kept is SegFormer3D adapted to anisotropic stacks (1 × 2 × 2 patches, no stage reduces the slices);
in its published configuration, SegFormer3D reaches an instance F1 of 0.06 on the same frame. Against convolutional
networks trained with the same recipe, ProSegFormer's advantage lies in the boundaries of the puncta (F1 at IoU 0.7),
not in finding them; it comes from the full-resolution network and the pretraining, not from self-attention.

The released checkpoints are the seed of each model chosen on validation (seed 12); see
[checkpoints/README.md](checkpoints/README.md) for their own test scores.

## Install

```bash
git clone https://github.com/ramugautam1/prosegformer.git
cd prosegformer
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Python 3.10 or newer. A GPU makes training practical (about 1.5 hours per model on one GPU); segmentation also runs on a
CPU.

## Segment a movie

```bash
prosegformer-segment --image movie.tif --channel 0 --out results/movie
```

- `--image`: a 3D stack (Z x Y x X), a movie (T x Z x Y x X), a TIFF with a channel axis (`--channel`), a NIfTI file
  or a folder of `t###.tif`; every time point is normalized and segmented on its own
- `--checkpoint`: `checkpoints/prosegformer.pt` (default) or `checkpoints/prosegformer_lite.pt`
- outputs in `--out`:
  - `probability.tif`: foreground probability (8-bit)
  - `mask.tif`: the thresholded foreground
  - `labels.tif`: puncta labelled 1..n in every time point
  - `objects.csv`: centroid (µm), volume and mean intensity of every punctum
  - `summary.json`: settings and counts
- objects are 26-connected components of at least 4 voxels; `--connectivity 6 --min-size 6` keeps puncta that touch
  only across sections apart, and gives the objects that [ProTrackFormer](https://github.com/ramugautam1/protrackformer)
  tracks

The voxel size is read from the TIFF metadata; set it with `--voxel-size Z Y X` (µm) otherwise. The model was trained on
stacks with 0.124 µm pixels and 0.5 µm sections; data with a very different sampling, or another protein, is best
segmented after fine-tuning (see [COMMANDS.md](COMMANDS.md)).

## Python

```python
from prosegformer.inference import Segmenter
from prosegformer.io import read_stack

movie = read_stack("movie.tif", channel=0)        # T x Z x Y x X
seg = Segmenter("checkpoints/prosegformer.pt")
probability, labels = seg(movie[0])                # one time point
```

## Checkpoints

| file | model | trained on |
|---|---|---|
| `checkpoints/prosegformer.pt` | ProSegFormer | Bazooka frames and crops, from the pretrained encoder |
| `checkpoints/prosegformer_lite.pt` | ProSegFormer-Lite | the same |
| `checkpoints/prosegformer_pretrained.pt` | ProSegFormer encoder | masked-image pretraining on six unannotated Bazooka movies |
| `checkpoints/prosegformer_lite_pretrained.pt` | ProSegFormer-Lite encoder | the same |

## Data

- `data/bazooka`: 8 annotated frames (13 x 320 x 512) and 28 tracking crops (13 x 32 x 32, 40 frames each)
- `data/gish`: 7 annotated stacks of Gilgamesh puncta, with 3 cross-validation folds
- `data/ecad`: 5 annotated stacks of E-cadherin puncta, with 3 cross-validation folds

See [data/README.md](data/README.md) for the layout and the annotation conventions.

## Train and evaluate

Every command, from training and pretraining to fine-tuning on another protein, is in [COMMANDS.md](COMMANDS.md).

```bash
prosegformer-train configs/prosegformer.yaml --seed 12 --out runs/prosegformer
prosegformer-evaluate --checkpoint runs/prosegformer/model.pt --data data/bazooka --split test
```

## Repository layout

```
configs/                 training and pretraining configurations (YAML)
data/                    Bazooka, Gish and E-cadherin segmentation ground truth
checkpoints/             trained and pretrained models
src/prosegformer/
    model.py             the network
    data.py              annotated volumes and training samples
    train.py             prosegformer-train
    pretrain.py          prosegformer-pretrain (masked-image pretraining)
    inference.py         tiled prediction with mirror averaging, and the Segmenter class
    metrics.py           instance, detection and voxel scores
    evaluate.py          prosegformer-evaluate
    segment.py           prosegformer-segment
    io.py                reading stacks and movies, writing ImageJ hyperstacks
    config.py            YAML configurations
tests/
```

## Citation

If you use ProSegFormer or its data, please cite the paper (see [CITATION.cff](CITATION.cff)).

## License

- code: MIT, see [LICENSE](LICENSE)
- data (`data/`): CC BY 4.0, see [data/LICENSE](data/LICENSE); please cite the paper when you use them
