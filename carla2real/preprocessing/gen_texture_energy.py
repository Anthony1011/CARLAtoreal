#!/usr/bin/env python3
"""Roughness prior: tell the generator HOW MUCH fine texture a surface carries, not what it looks like.

THE FAILURE THIS TARGETS. Mapillary class 30 means "vegetation" and nothing else. The generator has
to invent where every leaf is, and being a per-frame image model it invents differently each frame.
The measured consequence is the largest quality gap in the project (veg_report.py):

    real training photos   near 1191   far 1246   far/near 1.05
    CARLA source           near  928   far 1030   far/near 1.11
    v50 render             near  672   far  561   far/near 0.83

The render reaches 56% of the detail real photographs carry, and it is the only source that LOSES
detail with distance -- the further away a tree is, the less the model bothers. Two attempts to fix
this by re-weighting the loss (v63, v64) both raised distant foliage and both wrecked the road, from
opposite loss geometries. The information simply is not in the input.

WHY A CHANNEL AND NOT A LOSS. Every durable gain in this project came from giving the generator
something it did not have -- edges (v17 buildings, v19 walls, v44 vehicles), chroma (v50) -- and
every attempt to squeeze it out of the loss failed. This is the same move: CARLA renders real tree
geometry, so at inference the answer exists; it was simply never handed over.

WHY COARSE, AND WHY THIS IS NOT THE EDGE CHANNEL. The edge channel is sparse, binary and structural:
where does an object stop. This is continuous and areal: how rough is the surface here. Critically it
is computed at BLOCK resolution and blurred, so it cannot be copied as detail -- it says "this region
is 3x rougher than that one", not "put a leaf at this pixel". A per-pixel high-pass would let the
generator pass the channel through and would be a different image at inference (CARLA's own high
frequencies) than in training (the photo's), which is exactly the train/test mismatch that sank v65.

At training the map comes from the real photograph; at inference from the CARLA render. Same
convention as the edge channel, which has worked since v17.

CALIBRATE THE INFERENCE SIDE (--match-ref). A conditioning channel only helps if its DISTRIBUTION
matches between training and inference; otherwise the model extrapolates, and extrapolation is what
produced v65's crosshatch. Measured here, raw CARLA roughness is systematically higher than the
photographs the model learned from:

    training photos   mean 0.288   p25 0.067   p50 0.180   p75 0.431
    CARLA, raw        mean 0.394   p25 0.220   p50 0.322   p75 0.545

The median is nearly double. v66, trained on the photo distribution and shown the CARLA one, moved
road luminance +50 and vehicles -19 -- it was told the whole scene was rougher than any scene in
training and redistributed brightness accordingly. Luminance and roughness are uncorrelated in the
corpus (r = -0.08), so this is not the channel smuggling brightness; it is a distribution shift.

--match-ref histogram-matches the output to a reference set of maps, so what the model receives at
inference has the statistics it was trained on. RELATIVE roughness -- vegetation still reads far
rougher than road -- is preserved, because histogram matching is monotonic. Only the scale changes.

  usage: gen_texture_energy.py <image_dir> <out_dir> [--carla-bgr] [--ext jpg|png]
                               [--match-ref <dir_of_reference_maps>]
"""
import os
import sys

import cv2
import numpy as np

SRC, OUT = sys.argv[1], sys.argv[2]
CARLA_BGR = '--carla-bgr' in sys.argv          # CARLA writes BGRA; [:, :, :3] is BGR-as-RGB
EXT = sys.argv[sys.argv.index('--ext') + 1] if '--ext' in sys.argv else None
BLOCK = 16          # px; the map is one value per 16x16 tile before upsampling
SIGMA = 1.6         # high-pass cut, matching the measurement tools
CLIP = 12.0         # roughness above this is rare and would flatten the useful range

MATCH_REF = sys.argv[sys.argv.index('--match-ref') + 1] if '--match-ref' in sys.argv else None

os.makedirs(OUT, exist_ok=True)
exts = ('.jpg', '.jpeg', '.png') if EXT is None else ('.' + EXT,)
files = sorted(f for f in os.listdir(SRC) if f.lower().endswith(exts))
if not files:
    raise SystemExit(f'no images in {SRC}')

def cdf_from(dirpath, stride=200, cap=150):
    """256-bin CDF of a reference map set, for histogram matching."""
    fs = sorted(g for g in os.listdir(dirpath) if g.lower().endswith('.png'))[::stride][:cap]
    hist = np.zeros(256, np.float64)
    for g in fs:
        a = cv2.imread(os.path.join(dirpath, g), cv2.IMREAD_GRAYSCALE)
        if a is not None:
            hist += np.bincount(a[::4, ::4].ravel(), minlength=256)
    if hist.sum() == 0:
        raise SystemExit(f'no reference maps read from {dirpath}')
    c = np.cumsum(hist) / hist.sum()
    return c, len(fs)


LUT = None
if MATCH_REF:
    ref_cdf, nref = cdf_from(MATCH_REF)
    print(f'  matching to {nref} reference maps from {MATCH_REF}')

n = 0
vals = []
raw_hist = np.zeros(256, np.float64)
pending = []
for f in files:
    img = cv2.imread(os.path.join(SRC, f))
    if img is None:
        continue
    if CARLA_BGR:
        img = img[:, :, ::-1]
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), SIGMA))
    H, W = hp.shape
    # block mean, then bilinear back up: a coarse, smooth roughness field
    bh, bw = max(1, H // BLOCK), max(1, W // BLOCK)
    small = cv2.resize(hp, (bw, bh), interpolation=cv2.INTER_AREA)
    small = np.clip(small / CLIP, 0, 1)
    up = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    up = cv2.GaussianBlur(up, (0, 0), BLOCK / 2.0)
    vals.append(float(up.mean()))
    out = (up * 255.0).astype(np.uint8)
    if MATCH_REF:
        # two passes: the source CDF is only known once every frame has been seen, and matching
        # per-frame would erase real differences between frames
        raw_hist += np.bincount(out[::4, ::4].ravel(), minlength=256)
        pending.append((os.path.splitext(f)[0] + '.png', out))
    else:
        cv2.imwrite(os.path.join(OUT, os.path.splitext(f)[0] + '.png'), out)
    n += 1

if MATCH_REF and pending:
    src_cdf = np.cumsum(raw_hist) / raw_hist.sum()
    LUT = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)
    for name, arr in pending:
        cv2.imwrite(os.path.join(OUT, name), LUT[arr])
    q = [5, 25, 50, 75, 95]
    before = np.interp(np.array(q) / 100.0, src_cdf, np.arange(256)) / 255.0
    after = np.interp(np.array(q) / 100.0, np.cumsum(np.bincount(LUT[pending[0][1]].ravel(),
                      minlength=256)) / pending[0][1].size, np.arange(256)) / 255.0
    print('  histogram matched. percentiles before -> target:')
    print('    ' + '  '.join(f'p{a}: {b:.3f}' for a, b in zip(q, before)))

print(f'  wrote {n} roughness maps to {OUT}  (block {BLOCK}, clip {CLIP})')
print(f'  mean roughness {np.mean(vals):.3f}  (0 = flat everywhere, 1 = saturated)')
