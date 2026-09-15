#!/usr/bin/env python3
"""Flatten PandaSet's front camera into a flat, index-ordered image directory.

Written as a FILE rather than a heredoc because `conda run` silently discards stdin: the first
attempt at this ran in zero seconds, produced nothing, and reported no error. The corpus gate then
passed -- every channel had a consistent 23,406 entries -- because consistency across channels says
nothing about whether the new data arrived. Only train_v68's separate "is this corpus big enough"
check caught it.

Ordering is scene-then-frame, numerically. PandaSet frames are named 00.jpg..79.jpg, so a plain
lexical sort would put 10 before 2 and scramble the sequences that the temporal corpus builder
depends on.

Output is 1024x512 to match the three sources already in the corpus; every downstream generator
then runs on the same geometry.
"""
import os
import sys

from carla2real.config import DATA, ROOT, OUT  # noqa: E402

import cv2

PS = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(DATA, 'pandaset/extracted/pandaset')
OUT = sys.argv[2] if len(sys.argv) > 2 else \
    os.path.join(DATA, 'pandaset_stage/train_img')

os.makedirs(OUT, exist_ok=True)

files = []
for scene in sorted(os.listdir(PS)):
    d = os.path.join(PS, scene, 'camera', 'front_camera')
    if not os.path.isdir(d):
        continue
    names = [f for f in os.listdir(d) if f.lower().endswith('.jpg')]
    names.sort(key=lambda x: int(os.path.splitext(x)[0]))
    files += [os.path.join(d, f) for f in names]

if not files:
    raise SystemExit(f'no front_camera images under {PS}')

n = 0
for p in files:
    im = cv2.imread(p)
    if im is None:
        continue
    im = cv2.resize(im, (1024, 512), interpolation=cv2.INTER_AREA)
    cv2.imwrite(os.path.join(OUT, f'pshr_{n:05d}.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    n += 1

print(f'  wrote {n} PandaSet frames at 1024x512 to {OUT}')
if n < 8000:
    raise SystemExit(f'expected ~8240 frames, got {n} -- refusing to report success')
