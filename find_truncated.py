#!/usr/bin/env python3
"""Find image files that open but fail to fully decode.

WHY THIS EXISTS. MoGe's depth/normal pass was killed mid-write on 09-03 when the staging job and
the autopilot raced for the GPU. When it was resumed on 09-08 it skipped every output that already
existed -- and the file it had been writing at the moment it died existed, at zero or partial
length. Training then read 31,646 pairs for 1h11 before hitting it:

    OSError: image file is truncated   (aligned_dataset.py, normal_paths[index])

An existence check is not a validity check. The same lesson as the corpus gate that counted entries
without looking at what they were.

verify() alone is not enough: it checks structure without decoding, and a file truncated inside the
pixel data passes it. So this forces a real load().

  usage: find_truncated.py <dir> [<dir>...] [--delete]
"""
import os
import sys

from PIL import Image

DELETE = '--delete' in sys.argv
dirs = [a for a in sys.argv[1:] if not a.startswith('--')]
bad = []

for d in dirs:
    if not os.path.isdir(d):
        continue
    names = sorted(os.listdir(d))
    for i, n in enumerate(names):
        p = os.path.join(d, n)
        if os.path.getsize(p) == 0:
            bad.append((p, 'zero length'))
            continue
        try:
            with Image.open(p) as im:
                im.load()          # a real decode, not verify()
        except Exception as e:
            bad.append((p, f'{type(e).__name__}: {e}'))
        if (i + 1) % 5000 == 0:
            print(f'  {d}: {i + 1}/{len(names)}  bad so far {len(bad)}', flush=True)
    print(f'  {d}: {len(names)} files checked', flush=True)

print(f'\n{len(bad)} unreadable file(s)')
for p, why in bad[:40]:
    print(f'  {p}  --  {why}')
if DELETE:
    for p, _ in bad:
        os.remove(p)
    print(f'deleted {len(bad)} -- regenerate them, then re-run this to confirm clean')
