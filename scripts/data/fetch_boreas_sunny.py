#!/usr/bin/env python3
"""Download sunny-summer Boreas camera clips (CC BY 4.0, pyboreas DATA_LICENSE.md) for the colour fine-tune.

Per sequence, NS scenes of 80 consecutive frames (10 Hz, like PandaSet) spread evenly across the drive. Public S3 bucket,
no account. Each frame is cropped to 2:1 above the bonnet (rows 416..1640 of 2048) and saved as 1920x960 JPEG, sorted
scene-then-frame (bshr_<seq><scene><frame>), so the temporal builder can group consecutive frames.
  usage: fetch_boreas_sunny.py <out_dir> [NS]
"""
import os, sys, re, urllib.request
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
OUT = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 12
SEQS = ['boreas-2021-07-27-14-43', 'boreas-2021-08-05-13-34', 'boreas-2022-08-05-12-59', 'boreas-2022-08-05-13-30']
os.makedirs(OUT, exist_ok=True)


def keys(seq):
    ks, tok = [], None
    while True:
        u = f'https://boreas.s3.amazonaws.com/?list-type=2&prefix={seq}/camera/&max-keys=1000' + (f'&continuation-token={urllib.parse.quote(tok)}' if tok else '')
        x = urllib.request.urlopen(u, timeout=60).read().decode()
        ks += re.findall(r'<Key>([^<]*\.png)</Key>', x)
        m = re.search(r'<NextContinuationToken>([^<]*)</NextContinuationToken>', x)
        if not m: return sorted(ks)
        tok = m.group(1)


def get(job):
    k, dst = job
    if os.path.exists(dst): return 1
    try:
        b = urllib.request.urlopen('https://boreas.s3.amazonaws.com/' + k, timeout=120).read()
        im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
        im = cv2.resize(im[416:1640], (1920, 960), interpolation=cv2.INTER_AREA)
        cv2.imwrite(dst, im, [cv2.IMWRITE_JPEG_QUALITY, 95]); return 1
    except Exception as e:
        print('fail', k, e, flush=True); return 0


import urllib.parse
jobs = []
for si, seq in enumerate(SEQS):
    ks = keys(seq); n = len(ks); print(seq, n, 'frames', flush=True)
    for s in range(NS):
        st = int((s + 0.5) * n / NS) - 40
        for f in range(80):
            jobs.append((ks[st + f], f'{OUT}/bshr_{si:01d}{s:02d}{f:03d}.jpg'))
with ThreadPoolExecutor(12) as ex: ok = sum(ex.map(get, jobs))
print(f'downloaded {ok}/{len(jobs)} frames -> {OUT}')
