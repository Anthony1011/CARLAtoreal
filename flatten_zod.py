#!/usr/bin/env python3
"""Flatten ZOD Frames into sunny and night image directories at 1024x512.

Written as a FILE, not a heredoc: `conda run` discards stdin, and the PandaSet flatten step that
was written as a heredoc ran in zero seconds, produced nothing, and reported success.

SELECTION. ZOD Frames is 100k images, far more than this corpus needs -- PandaSet contributes 8,240
sunny and 4,320 night, and matching that keeps the corpus balanced rather than letting one source
dominate. So this takes a STRIDED sample, not the first N: ZOD frame ids run in collection order,
so the first 10,000 would be a handful of drives in a couple of cities, while a stride spreads the
sample over every drive, season and country in the set.

Day/night is decided by MEASURED luminance, not by metadata -- the same rule build_pandaset_night.py
uses, and for the same reason: a dataset's "evening" label and "dark enough that street lamps
dominate the exposure" are not the same thing. Annotations were not downloaded anyway.

THE STRIDE IS MEASURED, NOT ASSUMED. The first run of this script inherited PandaSet's thresholds
and a fixed 3x oversample, and produced 56 night frames in its first 4,000 -- it would have run to
completion and failed its own gate. A random probe of 1,500 ZOD frames shows why: the luminance
distribution is nothing like PandaSet's, sitting at p50 94 with only 10.6% below 45, and the night
frames are clustered by drive rather than spread evenly, so a sequential scan hits day-heavy runs
first. So the probe runs FIRST, the night fraction is measured, and the stride is set from it with
margin. Night is the binding constraint -- day frames are 85% of the set and never scarce.

Frames between the two thresholds (twilight) are DROPPED. They are ~4% of ZOD and they are exactly
the frames that would teach a sunny model to render dusk and a night model to render daylight.

  usage: flatten_zod.py [--raw DIR] [--out DIR] [--sunny N] [--night N]
                        [--night-lum 45] [--day-lum 70]
"""
import os
import sys

from config import DATA, ROOT, OUT  # noqa: E402

import cv2
import numpy as np


def arg(flag, default, cast=str):
    return cast(sys.argv[sys.argv.index(flag) + 1]) if flag in sys.argv else default


RAW = arg('--raw', os.path.join(DATA, 'zod'))
OUT = arg('--out', os.path.join(DATA, 'zod_stage'))
WANT_S = arg('--sunny', 10000, int)
WANT_N = arg('--night', 5000, int)
NIGHT_LUM = arg('--night-lum', 45.0, float)
DAY_LUM = arg('--day-lum', 70.0, float)

SUN_DIR = os.path.join(OUT, 'sunny', 'train_img')
NGT_DIR = os.path.join(OUT, 'night', 'train_img')
os.makedirs(SUN_DIR, exist_ok=True)
os.makedirs(NGT_DIR, exist_ok=True)

# ZOD nests as single_frames/<id>/camera_front_dnat/<ts>.jpg. Globbing on 'camera' rather than the
# exact directory keeps this working if the layout shifts between releases.
files = []
for root, _dirs, names in os.walk(RAW):
    if 'camera' not in root.lower():
        continue
    for f in names:
        if f.lower().endswith(('.jpg', '.png')):
            files.append(os.path.join(root, f))
files.sort()
if len(files) < 1000:
    raise SystemExit(f'only {len(files)} images under {RAW} -- download incomplete, refusing to run')
print(f'  {len(files)} ZOD images found', flush=True)

def luminance(path, reduced=True):
    """Mean grey. Decoding at 1/4 scale is ~10x faster on ZOD's 8 MP frames and moves the mean by
    well under a level, which is immaterial against thresholds tens of levels apart."""
    im = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4 if reduced else cv2.IMREAD_COLOR)
    return None if im is None else float(np.mean(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)))


# Probe first: measure what fraction of this dataset is actually dark enough to count as night,
# then size the candidate pool from that rather than from an assumption.
import random
random.seed(0)
probe = random.sample(files, min(1500, len(files)))
plums = [x for x in (luminance(p) for p in probe) if x is not None]
frac_night = sum(1 for x in plums if x < NIGHT_LUM) / max(len(plums), 1)
print(f'  probe: {len(plums)} frames, p50 {np.percentile(plums, 50):.0f}, '
      f'{100 * frac_night:.1f}% below {NIGHT_LUM:.0f}', flush=True)
if frac_night < 0.01:
    raise SystemExit(f'only {100 * frac_night:.1f}% of ZOD is below {NIGHT_LUM} -- the night '
                     f'threshold is wrong for this dataset, not the data')

# 1.6x margin on the night yield, because the night frames cluster by drive: a sequential scan can
# run well below the global rate for thousands of frames at a stretch.
need = min(len(files), int(WANT_N * 1.6 / frac_night))
stride = max(1, len(files) // need)
cand = files[::stride]
print(f'  need ~{need} candidates for {WANT_N} night -> stride {stride}, '
      f'{len(cand)} candidates', flush=True)

ns = nn = 0
for i, p in enumerate(cand):
    if ns >= WANT_S and nn >= WANT_N:
        break
    # decide on the cheap reduced decode, and only pay for the full read on a keeper -- roughly
    # two thirds of what is scanned is discarded
    lum = luminance(p)
    if lum is None:
        continue
    if lum < NIGHT_LUM and nn < WANT_N:
        dst, name, nn = NGT_DIR, f'zngt_{nn:05d}.jpg', nn + 1
    elif lum >= DAY_LUM and ns < WANT_S:
        dst, name, ns = SUN_DIR, f'zshr_{ns:05d}.jpg', ns + 1
    else:
        continue          # twilight, or a bucket already full
    im = cv2.imread(p)
    if im is None:
        continue
    im = cv2.resize(im, (1024, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(dst, name), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if (i + 1) % 2000 == 0:
        print(f'  scanned {i + 1}/{len(cand)}  sunny {ns}  night {nn}', flush=True)

print(f'  wrote {ns} sunny -> {SUN_DIR}')
print(f'  wrote {nn} night -> {NGT_DIR}')
if ns < 3000 or nn < 1500:
    raise SystemExit(f'too few frames (sunny {ns}, night {nn}) -- refusing to report success')
