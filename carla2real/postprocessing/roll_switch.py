#!/usr/bin/env python3
"""Use a fallback render while the CAMERA ROLLS (our own car flipping), the main render otherwise.

WHY. 2026-11-14: the lead sunny recipe (v98cb model) reads 'our car flipping' worse than the older v98sc recipe (VP .68 vs
.86); 45 of 47 differing frames come after the crash (frame 126), i.e. while the camera rotates. Detect that from CARLA's
own frames: per-frame in-plane rotation from a similarity fit to sparse optical flow; ROLL = |cumulative rotation| (deg).
Weight of the fallback ramps 0 -> 1 over ROLL_LO..ROLL_HI degrees, smoothed over SMOOTH frames, so normal driving (no roll)
is untouched and the switch is a short crossfade, not a cut.
  usage: roll_switch.py <main.mp4> <fallback.mp4> <carla_rgb_dir> <out>   env: ROLL_LO=20 ROLL_HI=40 SMOOTH=7 (normal driving drifts <5 deg, a car flipping ahead ~16; our own flip reaches 350)
"""
import glob, os, sys
import cv2, numpy as np

from carla2real.common.vidcodec import fourcc_for
LO, HI, SM = float(os.environ.get('ROLL_LO', '20')), float(os.environ.get('ROLL_HI', '40')), int(os.environ.get('SMOOTH', '7'))
MAIN, FB, RGB, OUT = sys.argv[1:5]
rgbs = sorted(glob.glob(os.path.join(RGB, '*.png')))
g = lambda p: cv2.cvtColor(cv2.resize(cv2.imread(p), (640, 320), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
rot = [0.0]; prev = g(rgbs[0])
for p in rgbs[1:]:
    cur = g(p); a = 0.0
    pts = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8)
    if pts is not None and len(pts) > 20:
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
        ok = st.ravel() == 1
        if ok.sum() > 20:
            M, _ = cv2.estimateAffinePartial2D(pts[ok], nxt[ok], method=cv2.RANSAC)
            if M is not None: a = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    rot.append(a); prev = cur
roll = np.abs(np.cumsum(rot))
w = np.clip((roll - LO) / (HI - LO), 0, 1)
w = np.convolve(np.pad(w, SM, mode='edge'), np.ones(2 * SM + 1) / (2 * SM + 1), mode='same')[SM:-SM]
print(f'roll max {roll.max():.1f} deg; fallback weight > 0.5 on {(w > 0.5).sum()} of {len(w)} frames (first {int(np.argmax(w > 0.5)) if (w > 0.5).any() else -1})')
a, b = cv2.VideoCapture(MAIN), cv2.VideoCapture(FB); W, H = int(a.get(3)), int(a.get(4))
vw = cv2.VideoWriter(OUT, fourcc_for(OUT), a.get(5) or 30, (W, H)); n = 0
while True:
    ok1, f1 = a.read(); ok2, f2 = b.read()
    if not ok1: break
    k = w[n] if n < len(w) else 0.0
    if ok2 and k > 0: f1 = cv2.addWeighted(f1, 1 - k, cv2.resize(f2, (W, H)), k, 0)
    vw.write(f1); n += 1
vw.release(); print(f'wrote {OUT}: {n} frames')
