#!/usr/bin/env python3
"""Add staged clips <clip>_<frame:3> (e.g. bm0103_042) to a temporal corpus as <PFX><clip>_<frame:06d> + Farneback flow.

Generic version of build_temporal_boreas.py for fetch_boreas_more.py stems; a clip is kept only if all 80 frames and every
conditioning channel exist. Flow t -> t-1 on grey at 512x256, float16 (same convention as the PandaSet builder).
  usage: build_temporal_clips.py <stage_dir> <corpus_dir> <prefix>
"""
import glob, os, sys, cv2, numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
SRC, DST, PFX = sys.argv[1:4]
CH = {'train_img': '.jpg', 'train_label': '.png', 'train_edge': '.png', 'train_depth': '.png', 'train_normal': '.png',
      'train_chroma': '.png', 'train_texture': '.png'}
for c in list(CH) + ['train_flow']: os.makedirs(f'{DST}/{c}', exist_ok=True)
clips = defaultdict(list)
for p in sorted(glob.glob(f'{SRC}/train_img/*.jpg')):
    s = os.path.basename(p)[:-4]; clips[s.rsplit('_', 1)[0]].append(s)


def grey(stem):
    return cv2.cvtColor(cv2.resize(cv2.imread(f'{SRC}/train_img/{stem}.jpg'), (512, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def clip(item):
    cid, fs = item
    if len(fs) < 80 or any(not os.path.exists(f'{SRC}/{c}/{f}{e}') for f in fs for c, e in CH.items()): return 0
    prev = None
    for k, f in enumerate(sorted(fs)):
        st = f'{PFX}{cid}_{k:06d}'
        for c, e in CH.items():
            d = f'{DST}/{c}/{st}{e}'
            if not os.path.lexists(d): os.symlink(os.path.realpath(f'{SRC}/{c}/{f}{e}'), d)
        g = grey(f)
        if prev is not None:
            np.save(f'{DST}/train_flow/{st}.npy', cv2.calcOpticalFlowFarneback(g, prev, None, 0.5, 3, 21, 3, 5, 1.2, 0).astype(np.float16))
        prev = g
    return 1


with ProcessPoolExecutor(16) as ex: kept = sum(ex.map(clip, clips.items()))
print(f'added {kept}/{len(clips)} clips ({kept * 80} frames) to {DST}')
