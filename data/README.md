# Segmentation ground truth

Hand-annotated 3D puncta of three junctional proteins in living *Drosophila* embryos, imaged on a Zeiss LSM 800
confocal microscope with GaAsP detectors. Licensed under CC BY 4.0 ([LICENSE](LICENSE)): free to use and adapt with
credit to the paper.

| | Bazooka | Gish | E-cadherin |
|---|---|---|---|
| protein | Par-3/Bazooka-GFP | Gilgamesh-GFP | E-cadherin-GFP |
| frames (single time points) | 8 of 13 x 320 x 512 | 7 of 14 x 512 x 512 | 5 of 13 x 280 x 512 |
| tracking crops | 28 of 13 x 32 x 32, 40 frames each | -- | -- |
| voxel (z, y, x) | 0.5, 0.124, 0.124 µm | 0.5, 0.132, 0.132 µm | 0.5, 0.124, 0.124 µm |
| partition | train / val / test | 3 cross-validation folds | 3 cross-validation folds |

## Layout

```
bazooka/
    meta.json                  voxel size, descriptions, original frame names
    splits.json                {"train": {"frames": [...], "crops": [...]}, "val": ..., "test": ...}
    frames/images/frame0.tif   raw stack, Z x Y x X, uint16
    frames/labels/frame0.tif   objects, Z x Y x X, 0 = background
    crops/images/Crop01.tif    raw crop movie, T x Z x Y x X
    crops/labels/Crop01.tif    objects in every frame, T x Z x Y x X
gish/, ecad/
    meta.json                  also the original stack names
    splits.json                {"folds": [{"train": [...], "val": [...], "test": [...]}, ...]}
    frames/images/, frames/labels/
```

## Annotation conventions

- frames: binary masks of whole fields of view, drawn in ImageJ; their objects are the 26-connected components of the
  mask (voxels that share a face, an edge or a corner)
- crops: instance masks drawn one punctum at a time, for the
  [ProTrackFormer](https://github.com/ramugautam1/protrackformer) tracking benchmark; labels are track identities, so
  touching puncta keep separate labels
- the two kinds were drawn differently, so ProSegFormer has one output for each; Gish and E-cadherin stacks are frames
- the smallest puncta were not annotated (first percentile of size: 4 voxels on the frames); evaluation ignores
  predicted objects smaller than 4 voxels
- the Gish masks leave out about a third of the brightest spots, and the E-cadherin masks are drawn loosely, so models trained on Bazooka transfer to them only after fine-tuning

## Partitions

- Bazooka frames: frames 0-5 train, 6 validation, 7 test (one movie, another embryo than the crops)
- Bazooka crops: the fixed split of the tracking benchmark, 16 train, 4 validation, 8 test
- Gish and E-cadherin: fold k tests one chunk of the stacks, validates on one stack of the next chunk and trains on
  the rest; every stack is tested once

## Loading

```python
from prosegformer.data import load_split

vols = load_split("data/bazooka", "test")          # frames and every crop frame (Crop04_t0, ...)
vols = load_split("data/gish", "test", fold=0)
v = vols[0]
v.name, v.kind, v.image.shape, v.labels.max()      # image normalized as during training
```
