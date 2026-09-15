#!/usr/bin/env python3
"""Build a three-way side-by-side comparison mp4 from three delivered clips.

Same idea as make_compare.py, extended to three panels so a baseline and two candidates can be
judged in one pass rather than by flicking between two files.

Each source is 1920x960, halved to 960x480 -- the SAME per-panel size make_compare.py uses, so
detail is directly comparable to the two-way clips already built. Three of those sit in a 2880-wide
frame, which is wider than 1080p but scales fine in any player and keeps the resolution where it
matters.

Frame N of each panel is shown against frame N of the others, which is meaningful here because all
three are rendered from the SAME recording -- identical camera path, identical traffic -- so any
difference on screen is the model, not the drive.

Panels are truncated to the shortest input rather than padded: a short clip would otherwise freeze
on its last frame while the others keep moving, which reads as a model artefact rather than as
missing footage.

  usage: make_compare3.py <a.mp4> <b.mp4> <c.mp4> <out.mp4> <label a> <label b> <label c>
"""
import sys

import cv2
import numpy as np

A, B, C, OUT = sys.argv[1:5]
LABS = (sys.argv[5] if len(sys.argv) > 5 else 'a',
        sys.argv[6] if len(sys.argv) > 6 else 'b',
        sys.argv[7] if len(sys.argv) > 7 else 'c')

caps = [cv2.VideoCapture(p) for p in (A, B, C)]
for p, c in zip((A, B, C), caps):
    if not c.isOpened():
        raise SystemExit(f'cannot open {p}')
fps = caps[0].get(cv2.CAP_PROP_FPS) or 30
W, H, BAR = 960, 480, 42
out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (W * 3, H + BAR))

n = 0
while True:
    frames = []
    for c in caps:
        ok, f = c.read()
        if not ok:
            break
        frames.append(cv2.resize(f, (W, H)))
    if len(frames) < 3:
        break
    canvas = np.zeros((H + BAR, W * 3, 3), np.uint8)
    for i, f in enumerate(frames):
        canvas[BAR:, i * W:(i + 1) * W] = f
        cv2.putText(canvas, LABS[i], (i * W + 14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85,
                    (255, 255, 255), 2)
        # dividers, so the joins are unambiguous on a dark night scene
        if i:
            cv2.line(canvas, (i * W, BAR), (i * W, H + BAR), (90, 90, 90), 2)
    cv2.putText(canvas, f'frame {n}', (W * 3 - 190, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (160, 160, 160), 2)
    out.write(canvas)
    n += 1

out.release()
for c in caps:
    c.release()
print(f'  wrote {OUT}  ({n} frames, {W*3}x{H+BAR})')
