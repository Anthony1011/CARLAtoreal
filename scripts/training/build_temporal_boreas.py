#!/usr/bin/env python3
"""Add Boreas sunny clips (CC BY 4.0) to a temporal corpus: b<seq><scene>_<frame> stems, Farneback t -> t-1 flow.

boreas_stage/train_* holds bshr_<seq:1><scene:2><frame:3> (fetch_boreas_sunny.py: 80 consecutive 10 Hz frames per scene)
with every conditioning channel. Same flow convention as build_temporal_pandaset.py (backward, grey at 512 wide, float16).
  usage: build_temporal_boreas.py <stage_dir> <corpus_dir>
"""
import glob, os, sys, cv2, numpy as np
from concurrent.futures import ProcessPoolExecutor
SRC, DST = sys.argv[1:3]
CH = {'train_img': '.jpg', 'train_label': '.png', 'train_edge': '.png', 'train_depth': '.png', 'train_normal': '.png',
      'train_chroma': '.png', 'train_texture': '.png'}
for c in list(CH) + ['train_flow']: os.makedirs(f'{DST}/{c}', exist_ok=True)
stems = sorted(os.path.basename(p)[:-4] for p in glob.glob(f'{SRC}/train_img/bshr_*.jpg'))
scenes = sorted({s[5:8] for s in stems})


def grey(stem):
    return cv2.cvtColor(cv2.resize(cv2.imread(f'{SRC}/train_img/{stem}.jpg'), (512, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def scene(sc):
    fs = [s for s in stems if s[5:8] == sc]
    if len(fs) < 80 or any(not os.path.exists(f'{SRC}/{c}/{f}{e}') for f in fs for c, e in CH.items()): return 0
    prev = None
    for k, f in enumerate(fs):
        st = f'b{sc}_{k:06d}'
        for c, e in CH.items():
            d = f'{DST}/{c}/{st}{e}'
            if not os.path.lexists(d): os.symlink(os.path.realpath(f'{SRC}/{c}/{f}{e}'), d)
        g = grey(f)
        if prev is not None:
            np.save(f'{DST}/train_flow/{st}.npy', cv2.calcOpticalFlowFarneback(g, prev, None, 0.5, 3, 21, 3, 5, 1.2, 0).astype(np.float16))
        prev = g
    return 1


with ProcessPoolExecutor(16) as ex: kept = sum(ex.map(scene, scenes))
print(f'added {kept}/{len(scenes)} Boreas scenes ({kept * 80} frames) to {DST}')
