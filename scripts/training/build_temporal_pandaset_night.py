#!/usr/bin/env python3
"""Licensed NIGHT sequence corpus for self-consistency training: PandaSet night clips (CC BY 4.0).

pandaset_night_stage/train_* holds psngt_00000..04319 = 18 night scenes x 3 forward cameras x 80 frames, written
scene -> camera -> frame (build_pandaset_night.py), so every 80 consecutive files are one clip. Linked as
n<clip:03d>_<frame:06d> so TemporalDataset groups them; Farneback t -> t-1 flow (grey at 512x256, float16).
"""
import os, cv2, numpy as np
from concurrent.futures import ProcessPoolExecutor
B = os.environ.get('CARLA2REAL_DATA', 'datasets'); SRC = f'{B}/pandaset_night_stage'; DST = f'{B}/training_temporal_pandaset_night'
CH = {'train_img': '.jpg', 'train_label': '.png', 'train_edge': '.png', 'train_depth': '.png', 'train_normal': '.png', 'train_light': '.png'}
for c in list(CH) + ['train_flow']: os.makedirs(f'{DST}/{c}', exist_ok=True)
N = len(os.listdir(f'{SRC}/train_img')) // 80


def grey(i):
    return cv2.cvtColor(cv2.resize(cv2.imread(f'{SRC}/train_img/psngt_{i:05d}.jpg'), (512, 256), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)


def clip(k):
    prev = None
    for f in range(80):
        i = k * 80 + f; st = f'n{k:03d}_{f:06d}'
        for c, e in CH.items():
            d = f'{DST}/{c}/{st}{e}'
            if not os.path.lexists(d): os.symlink(os.path.realpath(f'{SRC}/{c}/psngt_{i:05d}{e}'), d)
        g = grey(i)
        if prev is not None:
            np.save(f'{DST}/train_flow/{st}.npy', cv2.calcOpticalFlowFarneback(g, prev, None, 0.5, 3, 21, 3, 5, 1.2, 0).astype(np.float16))
        prev = g
    return 1


with ProcessPoolExecutor(16) as ex: kept = sum(ex.map(clip, range(N)))
print(f'{kept} night clips -> {kept * 80} frames in {DST}')
