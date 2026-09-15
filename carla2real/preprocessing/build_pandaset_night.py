#!/usr/bin/env python3
"""Extract PandaSet's night scenes as a replacement for the 3,600 video-derived night pairs.

THE SHORTFALL, and how it is closed. The night corpus is 6,826 pairs, of which ngt_ 3,600 come from
the four unlicensed night videos. PandaSet has 18 genuinely night scenes (mean luminance under 45),
but only 80 front-camera frames each -- 1,440, well short of 3,600.

The three FORWARD-FACING cameras close the gap: front, front_left and front_right are all real
street scenes from the same drive, and for image-to-image training the camera's yaw does not matter
-- the model learns what a lit street looks like at night, not what a particular mounting angle
sees. 18 x 80 x 3 = 4,320, comfortably over the 3,600 being replaced. The back and side cameras are
deliberately excluded: they look along or away from the kerb and their content distribution is not
what the CARLA forward camera will present at inference.

Night is selected by MEASURED luminance rather than by scene metadata, because "evening" in a
dataset description and "dark enough that street lamps dominate" are not the same thing.

  usage: build_pandaset_night.py [<zip>] [<out_dir>] [--lum 45]
"""
import os
import subprocess
import sys

from carla2real.config import DATA, ROOT, OUT  # noqa: E402
import tempfile

import cv2
import numpy as np

ZIP = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else \
    os.path.join(DATA, 'pandaset/pandaset.zip')
OUT = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else \
    os.path.join(DATA, 'pandaset_night_stage/train_img')
LUM = float(sys.argv[sys.argv.index('--lum') + 1]) if '--lum' in sys.argv else 45.0
CAMS = ('front_camera', 'front_left_camera', 'front_right_camera')
EXTRACTED = os.path.join(DATA, 'pandaset/extracted/pandaset')

os.makedirs(OUT, exist_ok=True)

# [1] find the night scenes from the already-extracted front camera
night = []
for scene in sorted(os.listdir(EXTRACTED)):
    d = os.path.join(EXTRACTED, scene, 'camera', 'front_camera')
    if not os.path.isdir(d):
        continue
    fs = sorted(f for f in os.listdir(d) if f.endswith('.jpg'))
    if not fs:
        continue
    vals = []
    for f in fs[::20][:4]:
        im = cv2.imread(os.path.join(d, f))
        if im is not None:
            vals.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).mean())
    if vals and np.mean(vals) < LUM:
        night.append(scene)
print(f'  {len(night)} night scenes (mean luminance < {LUM}): {" ".join(night)}')
if not night:
    raise SystemExit('no night scenes found')

# [2] pull the two extra forward cameras for those scenes only -- extracting all six for all 103
# scenes would be 50k images to throw most of away
patterns = [f'pandaset/{s}/camera/{c}/*' for s in night for c in CAMS[1:]]
with tempfile.TemporaryDirectory(dir=os.path.dirname(OUT)) as tmp:
    subprocess.run(['unzip', '-q', '-o', ZIP] + patterns + ['-d', tmp], check=False)

    n = 0
    for scene in night:
        for cam in CAMS:
            src = os.path.join(EXTRACTED, scene, 'camera', cam)
            if not os.path.isdir(src):
                src = os.path.join(tmp, 'pandaset', scene, 'camera', cam)
            if not os.path.isdir(src):
                continue
            names = [f for f in os.listdir(src) if f.lower().endswith('.jpg')]
            names.sort(key=lambda x: int(os.path.splitext(x)[0]))
            for f in names:
                im = cv2.imread(os.path.join(src, f))
                if im is None:
                    continue
                im = cv2.resize(im, (1024, 512), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(OUT, f'psngt_{n:05d}.jpg'), im,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1

print(f'  wrote {n} night frames at 1024x512 to {OUT}')
if n < 3600:
    raise SystemExit(f'only {n} night frames -- fewer than the 3,600 being replaced, refusing to '
                     f'report success')
