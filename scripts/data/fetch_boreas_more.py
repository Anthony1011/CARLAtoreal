#!/usr/bin/env python3
"""Download MORE daytime Boreas clips (CC BY 4.0, pyboreas DATA_LICENSE.md) for the big temporal colour/flicker retrain.

USER 11-12: "make more space for more video data" (disk cleared to 572 GB). Daytime, non-winter sequences; per sequence NS
scenes of 80 consecutive 10 Hz frames spread across the drive. A scene is skipped unless its middle frame is daylight
(mean grey >= DAY_MIN after the bonnet crop). Saved as bm<seq:2><scene:2>_<frame:3>.jpg, 1920x960 (rows 416..1640).
  usage: fetch_boreas_more.py <out_dir> [NS]   env: DAY_MIN=90
"""
import os, sys, re, urllib.request, urllib.parse
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
OUT = sys.argv[1]; NS = int(sys.argv[2]) if len(sys.argv) > 2 else 12; DAY = float(os.environ.get('DAY_MIN', '90'))
SEQS = ['boreas-2021-04-08-12-44', 'boreas-2021-04-13-14-49', 'boreas-2021-04-20-14-11', 'boreas-2021-04-22-15-00',
        'boreas-2021-04-29-15-55', 'boreas-2021-05-06-13-19', 'boreas-2021-05-13-16-11', 'boreas-2021-06-03-16-00',
        'boreas-2021-06-17-17-52', 'boreas-2021-07-20-17-33', 'boreas-2021-07-27-14-43', 'boreas-2021-08-05-13-34',
        'boreas-2021-09-02-11-42', 'boreas-2021-09-07-09-35', 'boreas-2021-09-09-15-28', 'boreas-2021-10-05-15-35',
        'boreas-2021-10-15-12-35', 'boreas-2021-10-22-11-36', 'boreas-2021-10-26-12-35', 'boreas-2022-08-05-12-59',
        'boreas-2022-08-05-13-30', 'boreas-2022-08-05-13-54', 'boreas-2022-08-05-15-01']
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


def fetch(k):
    b = urllib.request.urlopen('https://boreas.s3.amazonaws.com/' + k, timeout=120).read()
    im = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
    return cv2.resize(im[416:1640], (1920, 960), interpolation=cv2.INTER_AREA)


def get(job):
    k, dst = job
    if os.path.exists(dst): return 1
    try:
        cv2.imwrite(dst, fetch(k), [cv2.IMWRITE_JPEG_QUALITY, 95]); return 1
    except Exception as e:
        print('fail', k, e, flush=True); return 0


jobs, kept, skipped = [], 0, 0
for si, seq in enumerate(SEQS):
    try: ks = keys(seq)
    except Exception as e: print(seq, 'list failed', e, flush=True); continue
    n = len(ks); print(seq, n, 'frames', flush=True)
    if n < 400: continue
    for s in range(NS):
        st = int((s + 0.5) * n / NS) - 40
        try: g = cv2.cvtColor(fetch(ks[st + 40]), cv2.COLOR_BGR2GRAY).mean()
        except Exception: continue
        if g < DAY: skipped += 1; continue
        kept += 1
        for f in range(80): jobs.append((ks[st + f], f'{OUT}/bm{si:02d}{s:02d}_{f:03d}.jpg'))
print(f'{kept} daylight scenes kept, {skipped} too dark; downloading {len(jobs)} frames', flush=True)
with ThreadPoolExecutor(16) as ex: ok = sum(ex.map(get, jobs))
print(f'downloaded {ok}/{len(jobs)} frames -> {OUT}')
