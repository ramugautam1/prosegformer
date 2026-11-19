"""The ProSegFormer network.

- stem: two 3x3x3 convolutions at full resolution
- stage s: blocks of a depthwise-convolution position encoding, window self-attention and a convolutional
  feed-forward layer, at 1/2^s of the in-plane resolution; the depth axis is never downsampled
- windows span every slice and `window` x `window` pixels; a window of 0 means global attention
- decoder: transposed-convolution upsampling, skip connections and two 3x3x3 convolutions per level
- one output channel per annotation convention, and auxiliary outputs at 1/2 and 1/4 for deep supervision
- ProSegFormer-Lite is the same network without self-attention (`attention=(0, 0, 0)`)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT = {"dims": [32, 64, 128], "depths": [2, 2, 2], "heads": [2, 4, 8], "windows": [8, 8, 0],
           "attention": [1, 1, 1], "outputs": 2}
LITE = {**DEFAULT, "attention": [0, 0, 0]}


def conv_block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv3d(i, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True), nn.GELU(),
                         nn.Conv3d(o, o, 3, padding=1, bias=False), nn.InstanceNorm3d(o, affine=True), nn.GELU())


class WindowAttention(nn.Module):
    def __init__(self, dim: int, heads: int, window: int):
        super().__init__()
        self.h, self.w = heads, window
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, Z, H, W, C)
        B, Z, H, W, C = x.shape
        wh, ww = (min(self.w, H), min(self.w, W)) if self.w else (H, W)
        nh, nw = H // wh, W // ww
        t = x.view(B, Z, nh, wh, nw, ww, C).permute(0, 2, 4, 1, 3, 5, 6).reshape(B * nh * nw, Z * wh * ww, C)
        q, k, v = self.qkv(t).view(t.shape[0], -1, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        o = F.scaled_dot_product_attention(q, k, v)
        o = self.proj(o.transpose(1, 2).reshape(t.shape[0], -1, C))
        return o.view(B, nh, nw, Z, wh, ww, C).permute(0, 3, 1, 4, 2, 5, 6).reshape(B, Z, H, W, C)


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, mlp: int = 4, attention: bool = True):
        super().__init__()
        self.cpe = nn.Conv3d(dim, dim, 3, padding=1, groups=dim)
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, heads, window) if attention else None
        self.fc1, self.fc2 = nn.Linear(dim, mlp * dim), nn.Linear(mlp * dim, dim)
        self.dw = nn.Conv3d(mlp * dim, mlp * dim, 3, padding=1, groups=mlp * dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, Z, H, W)
        x = x + self.cpe(x)
        t = x.permute(0, 2, 3, 4, 1)
        if self.attn is not None:
            t = t + self.attn(self.n1(t))
        h = self.fc1(self.n2(t)).permute(0, 4, 1, 2, 3).contiguous()
        h = F.gelu(self.dw(h)).permute(0, 2, 3, 4, 1)
        t = t + self.fc2(h)
        return t.permute(0, 4, 1, 2, 3).contiguous()


class ProSegFormer(nn.Module):
    def __init__(self, dims=(32, 64, 128), depths=(2, 2, 2), heads=(2, 4, 8), windows=(8, 8, 0), outputs=2,
                 stem=None, deep_supervision=True, attention=(1, 1, 1)):
        super().__init__()
        stem = stem or dims[0]
        self.stem = conv_block(1, stem)
        self.embed = nn.ModuleList()
        self.stages = nn.ModuleList()
        c = stem
        for s, (d, n, h, w, a) in enumerate(zip(dims, depths, heads, windows, attention)):
            down = nn.Conv3d(c, d, 1) if s == 0 else nn.Conv3d(c, d, (1, 2, 2), stride=(1, 2, 2))
            self.embed.append(nn.Sequential(down, nn.InstanceNorm3d(d, affine=True)))
            self.stages.append(nn.Sequential(*[Block(d, h, w, attention=bool(a)) for _ in range(n)]))
            c = d
        levels = range(len(dims) - 1, 0, -1)
        self.up = nn.ModuleList([nn.ConvTranspose3d(dims[s], dims[s - 1], (1, 2, 2), stride=(1, 2, 2)) for s in levels])
        self.dec = nn.ModuleList([conv_block(2 * dims[s - 1], dims[s - 1]) for s in levels])
        self.fuse = conv_block(dims[0] + stem, dims[0])
        self.head = nn.Conv3d(dims[0], outputs, 1)
        self.ds = deep_supervision
        self.aux = (nn.ModuleList([nn.Conv3d(dims[s], outputs, 1) for s in range(1, len(dims))])
                    if deep_supervision else None)
        self.mask_token = nn.Parameter(torch.zeros(1, stem, 1, 1, 1))

    def forward(self, x: torch.Tensor, aux: bool = False, mask: torch.Tensor | None = None):
        """x: (B, 1, Z, H, W). `mask` (same shape, 0/1) replaces the stem features of masked voxels by a learned
        token, for masked-image pretraining. With `aux`, also returns the outputs at 1/4 and 1/2 resolution."""
        st = self.stem(x)
        if mask is not None:
            st = st * (1 - mask) + mask * self.mask_token.to(st.dtype)
        skips, h = [], st
        for e, s in zip(self.embed, self.stages):
            h = s(e(h))
            skips.append(h)
        outs = []
        h = skips.pop()
        for k, (u, d) in enumerate(zip(self.up, self.dec)):
            if aux and self.ds:
                outs.append(self.aux[len(self.aux) - 1 - k](h))
            h = d(torch.cat([u(h), skips.pop()], 1))
        y = self.head(self.fuse(torch.cat([h, st], 1)))
        return (y, outs[::-1]) if aux else y


def build_model(cfg: dict | None = None, deep_supervision: bool = True) -> ProSegFormer:
    c = {**DEFAULT, **(cfg or {})}
    return ProSegFormer(dims=tuple(c["dims"]), depths=tuple(c["depths"]), heads=tuple(c["heads"]),
                        windows=tuple(c["windows"]), outputs=c["outputs"], attention=tuple(c["attention"]),
                        deep_supervision=deep_supervision)


def size_multiple(cfg: dict | None = None) -> int:
    """In-plane sizes that a whole-volume forward pass accepts are multiples of this number."""
    c = {**DEFAULT, **(cfg or {})}
    m = 2 ** (len(c["dims"]) - 1)
    for s, w in enumerate(c["windows"]):
        if w:
            m = max(m, w * 2 ** s)
    return m
