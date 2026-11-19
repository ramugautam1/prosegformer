# Checkpoints

| file | model | outputs | trained on | thresholds (frame / crop) |
|---|---|---|---|---|
| `prosegformer.pt` | ProSegFormer, 1.2 M parameters | frame, crop | Bazooka frames 0-5 and 16 crops, from `prosegformer_pretrained.pt` | 0.35 / 0.45 |
| `prosegformer_lite.pt` | ProSegFormer-Lite, 1.0 M parameters | frame, crop | the same, from `prosegformer_lite_pretrained.pt` | 0.75 / 0.55 |
| `prosegformer_pretrained.pt` | ProSegFormer encoder | reconstruction | masked-image pretraining, six unannotated Bazooka movies | -- |
| `prosegformer_lite_pretrained.pt` | ProSegFormer-Lite encoder | reconstruction | the same | -- |

- the two trained models are the seed (of 10, 11 and 12) with the best validation score, the mean instance F1 at IoU
  0.5 over the validation frame and crops; seed 12 in both cases
- thresholds were chosen on validation with mirror averaging, which `prosegformer-segment` and `prosegformer-evaluate`
  use by default
- the frame output follows the whole-field binary masks and is the one to use on new data; the crop output follows the
  instance masks of the tracking crops
- the pretrained encoders only initialize training (`init:` in a configuration)

Test scores of the released models (mirror averaging; the three-seed means are in the main README):

| model | frame F1@0.5 | frame F1@0.7 | frame det. F1 | crop F1@0.5 | crop F1@0.7 | crop det. F1 | Gish det. F1 | E-cadherin det. F1 |
|---|---|---|---|---|---|---|---|---|
| `prosegformer.pt` | 0.879 | 0.804 | 0.901 | 0.881 | 0.816 | 0.885 | 0.226 | 0.670 |
| `prosegformer_lite.pt` | 0.893 | 0.815 | 0.900 | 0.876 | 0.806 | 0.879 | 0.228 | 0.692 |

Gish and E-cadherin are zero-shot (frame output, every stack). Their annotations follow other conventions, so
fine-tune before use on those proteins (see COMMANDS.md).

Each file holds the weights, the network settings, the thresholds and a short description:

```python
import torch

ck = torch.load("checkpoints/prosegformer.pt", weights_only=False)
ck["config"], ck["thresholds"], ck["description"]
```
