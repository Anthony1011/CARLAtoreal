#!/usr/bin/env python3
"""Make ZOD look like the sharp, colourful sources instead of down-weighting it.

THE PROBLEM. Measured per corpus source at a common 1024 width:

    source        detail    sat     licence
    PandaSet        5.57   42.7     CC BY 4.0
    Mapillary       5.43   61.8     research only
    ZOD             2.71   37.1     CC BY-SA 4.0

ZOD is half as sharp as the sharp sources and the least saturated, and it is 78% of
training_clean_sunny. v81 lost 41% of its detail and 27% of its vehicle saturation to that, and the
user named both defects on sight -- "the colours are worse, and the car details are more mushy".

v83 answers by weighting ZOD down to 38%, which works but discards 20,000 frames of diversity.
This answers by fixing ZOD, so every frame still counts.

SHARPENING WITHOUT RINGING. A plain unsharp mask strong enough to reach PandaSet's detail overshoots
at 26% of edge pixels, and a generator trained on ringing learns to paint ringing -- the same
failure as ZOD's --blur variant, where smeared anonymisation patches taught the model to paint blur.
So every pixel is clamped to the local min/max of the INPUT after sharpening, which makes overshoot
impossible by construction rather than merely small. Measured:

    amount  radius  clamp   ZOD detail   vs PandaSet   halo %
       0.5     1.2   none        3.89          0.73      7.52
       1.2     1.2   none        5.22          0.97     26.36
       2.5     1.2    5x5        4.83          0.90      0.00   <- used here

90% of PandaSet's sharpness at zero overshoot, confirmed by eye on zshr_04000: crisper branches,
window frames and number plates, no visible halo.

SATURATION. Lifted toward the midpoint of PandaSet and Mapillary rather than all the way to
Mapillary, because the user's complaint about v75q was the opposite -- its cars run at 2.10x CARLA's
saturation. In S only, in HSV, so hue is untouched.

TWO CHANGES AT ONCE, stated plainly: this alters sharpness and saturation together, so a win does
not say which mattered. They are deficits of the same source and the user named both, and there is
time for two models before Tuesday, not four. Ablate later if it wins.

  usage: fix_zod_appearance.py <src_img_dir> <dst_img_dir> [--prefix zshr_] [--amount 2.5] [--sat 1.30]
"""
import os
import sys

import cv2
import numpy as np

SRC, DST = sys.argv[1], sys.argv[2]


def arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


PREFIX = sys.argv[sys.argv.index('--prefix') + 1] if '--prefix' in sys.argv else 'zshr_'
AMOUNT = arg('--amount', 2.5)
RADIUS = arg('--radius', 1.2)
CLAMP = int(arg('--clamp', 5))
SAT = arg('--sat', 1.30)          # 37.1 * 1.30 = 48.2, between PandaSet 42.7 and Mapillary 61.8

os.makedirs(DST, exist_ok=True)
ker = np.ones((CLAMP, CLAMP), np.uint8)
files = sorted(f for f in os.listdir(SRC)
               if f.startswith(PREFIX) and f.lower().endswith(('.png', '.jpg', '.jpeg')))
if not files:
    raise SystemExit(f'no images matching {PREFIX!r} in {SRC}')
print(f'  {len(files)} frames, unsharp {AMOUNT}/{RADIUS} clamped {CLAMP}x{CLAMP}, sat x{SAT}')

d0, d1, s0, s1 = [], [], [], []
n = 0
for f in files:
    a = cv2.imread(os.path.join(SRC, f))
    if a is None:
        continue
    blur = cv2.GaussianBlur(a, (0, 0), RADIUS)
    out = cv2.addWeighted(a, 1 + AMOUNT, blur, -AMOUNT, 0).astype(np.float32)
    lo = cv2.erode(a, ker).astype(np.float32)
    hi = cv2.dilate(a, ker).astype(np.float32)
    out = np.clip(out, lo, hi).astype(np.uint8)

    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * SAT, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    cv2.imwrite(os.path.join(DST, f), out,
                [cv2.IMWRITE_JPEG_QUALITY, 97] if f.lower().endswith(('.jpg', '.jpeg')) else [])
    if n % 200 == 0:
        for img, dl, sl in ((a, d0, s0), (out, d1, s1)):
            g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
            dl.append(float(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.6)).mean()))
            sl.append(float(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 1].mean()))
    n += 1
    if n % 5000 == 0:
        print(f'  {n}/{len(files)}', flush=True)

print(f'  wrote {n} frames -> {DST}')
print(f'  detail {np.mean(d0):.2f} -> {np.mean(d1):.2f}   (PandaSet 5.57, Mapillary 5.43)')
print(f'  sat    {np.mean(s0):.1f} -> {np.mean(s1):.1f}   (PandaSet 42.7, Mapillary 61.8)')
