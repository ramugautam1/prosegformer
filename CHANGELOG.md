# Changelog

## 1.0.0

- ProSegFormer: window self-attention from full in-plane resolution, no downsampling in depth, one output per
  annotation convention, masked-image pretraining
- ProSegFormer-Lite: the same network without self-attention
- segmentation ground truth for Bazooka (8 frames, 28 tracking crops), Gish (7 stacks) and E-cadherin (5 stacks), with
  splits and cross-validation folds (CC BY 4.0)
- `prosegformer-segment`, `prosegformer-train`, `prosegformer-pretrain`, `prosegformer-evaluate`
- trained and pretrained checkpoints of both models
