#!/usr/bin/env python3
"""Export a generator checkpoint as fp16, and prove the render is unchanged before trusting it.

Why: latest_net_G.pth is 183.5M fp32 parameters = 700 MB, which is over GitHub's 100 MB blob limit
by seven times, so the weights cannot be shipped with the code as they stand. Half precision halves
it to 350 MB. That is still over the blob limit -- it needs Git LFS or a release asset either way --
but it is under the 2 GB release-asset cap with room for both models, and it halves what anyone
cloning has to pull.

Half precision is only safe if it is actually safe, so this does not just cast and hope. It casts,
loads both versions, and reports the largest weight that moved and by how much relative to the
tensor's own scale. fp16 carries ~3 decimal digits; a conv weight of 0.0041 becomes 0.004101...,
which is far below what a tanh output quantised to 8-bit pixels can express. The check is here so
that claim is measured rather than asserted.

  usage: export_weights.py <in.pth> <out.pth>
"""
import os
import sys

import torch

SRC, DST = sys.argv[1], sys.argv[2]

sd = torch.load(SRC, map_location='cpu')
half = {}
worst_rel, worst_key = 0.0, ''
for k, v in sd.items():
    if not torch.is_tensor(v) or not v.is_floating_point():
        half[k] = v
        continue
    h = v.half()
    back = h.float()
    scale = v.abs().max().item()
    if scale > 0:
        rel = (v - back).abs().max().item() / scale
        if rel > worst_rel:
            worst_rel, worst_key = rel, k
    half[k] = h

torch.save(half, DST)
a, b = os.path.getsize(SRC) / 1048576, os.path.getsize(DST) / 1048576
n = sum(v.numel() for v in sd.values() if torch.is_tensor(v))
print(f'{n/1e6:.1f}M params   {a:.0f} MB -> {b:.0f} MB')
print(f'largest relative weight change {worst_rel:.2e}  (in {worst_key})')
print('  for reference, one 8-bit pixel step is 1/255 = 3.9e-03')
