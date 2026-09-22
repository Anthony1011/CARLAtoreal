#!/usr/bin/env python3
"""Measure a corpus source on the three axes that have decided every recent experiment.

Built because "add more data" keeps being the obvious move and keeps being wrong for reasons only
visible in these numbers. The reference values, all measured at a common 1024 width:

    source        detail    sat   hard-sun share   licence
    PandaSet        5.57   42.7        10%         CC BY 4.0
    Mapillary       5.43   61.8        76%         research only
    Cityscapes      3.66   57.0         --         research only
    ZOD             2.71   37.1        10%         CC BY-SA 4.0
    old footage     1.98   60.7         --         unlicensed

A new source is worth its staging cost only if it beats ZOD on detail or Mapillary-like on sun.

  detail    mean high-pass energy at sigma 1.6, at a common width so this is sharpness of content
            rather than of resolution
  sat       mean HSV saturation
  hard sun  share of frames whose ground p05 sits below 0.35 of its 75th percentile -- the depth
            Mapillary's median frame carries, and the property the licensed corpus lacks

  usage: measure_source.py <img_dir> [--num 300] [--detail-only|--sun-only]
"""
import os
import sys

import cv2
import numpy as np

SRC = sys.argv[1]
N = int(sys.argv[sys.argv.index('--num') + 1]) if '--num' in sys.argv else 300
QUIET = '--detail-only' in sys.argv or '--sun-only' in sys.argv

files = sorted(f for f in os.listdir(SRC) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
if not files:
    raise SystemExit(f'no images in {SRC}')
files = files[::max(1, len(files) // N)][:N]

det, sat, sun = [], [], []
for f in files:
    a = cv2.imread(os.path.join(SRC, f))
    if a is None:
        continue
    h = max(1, int(a.shape[0] * 1024 / a.shape[1]))
    b = cv2.resize(a, (1024, h), interpolation=cv2.INTER_AREA)
    g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    det.append(float(np.abs(g - cv2.GaussianBlur(g, (0, 0), 1.6)).mean()))
    sat.append(float(cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 1].mean()))
    low = g[int(g.shape[0] * 0.55):]           # the bottom 45% is ground in a forward camera
    ref = np.percentile(low, 75)
    if ref >= 1:
        sun.append(float(np.percentile(low, 5) / ref))

d, s = float(np.mean(det)), float(np.mean(sat))
hard = float((np.array(sun) < 0.35).mean() * 100) if sun else float('nan')

if '--detail-only' in sys.argv:
    print(f'{d:.2f}')
elif '--sun-only' in sys.argv:
    print(f'{hard:.0f}')
else:
    print(f'  {len(det)} frames from {SRC}')
    print(f'  detail    {d:.2f}   (ZOD 2.71, Cityscapes 3.66, Mapillary 5.43, PandaSet 5.57)')
    print(f'  sat       {s:.1f}   (ZOD 37.1, PandaSet 42.7, Cityscapes 57.0, Mapillary 61.8)')
    print(f'  hard sun  {hard:.0f}%   (ZOD 10%, PandaSet 10%, Mapillary 76%)')
