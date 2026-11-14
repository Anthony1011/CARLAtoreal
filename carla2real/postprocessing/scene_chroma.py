#!/usr/bin/env python3
"""Give buildings, banners and walls their colour back: Lab a,b from CARLA, lightness and texture from the render.

WHY. User, 2026-11-03, on v98scvs: "it looks muddy, the banners on Town10 colours are all muddy and it looks like a
dystopian country ... make it more realistic but still make vision pilot score high". The generator (trained on
overcast PandaSet/ZOD streets) paints colourful facades and banners grey-olive. CARLA's frame is pixel-aligned with
the render (same camera, same recording), so its chroma can be borrowed exactly like sign_pass borrows sign faces --
but only a,b: L (exposure, shading, the real-photo texture) stays the render's, so it does not turn back into CARLA.
CARLA's chroma is scaled by GAIN (real photographs are less saturated than a game engine) and blended by STRENGTH.
Vehicles are untouched (car colours already come from CARLA in the vehicle pass); vegetation untouched (user wants
soft, not vivid trees).
  usage: scene_chroma.py <in.mp4> <carla_rgb_dir> <label_dir> <out> | --frame N <in> <rgb> <lab> <out.png>
  env: CLASSES=17,6,3 GAIN=0.8 STRENGTH=0.75 VEH_MARGIN=0 CMIN=0
  VEH_MARGIN (px at 1920 wide): leave the facade right behind/around vehicles as rendered, so the detector sees the
  same car-vs-background contrast as on v98scvs. CMIN: only recolour where CARLA itself is colourful (Lab chroma ramps
  in over CMIN..2*CMIN), i.e. banners and signage, not plain grey/beige walls.
  LUMA (0..1, added 11-06; user: "walls look unpainted, dull"): C5 facades sit at Lab L ~117 vs CARLA ~157. Move the
  facades' LOW-FREQUENCY lightness (Gaussian sigma LUMA_SIG px) toward CARLA's by LUMA, keeping the render's own fine
  texture; same classes and vehicle margin, not chroma-gated (plain walls are exactly the dull ones).
"""
import glob, os, sys
import cv2, numpy as np
from PIL import Image

from carla2real.common.vidcodec import fourcc_for
CLASSES = [int(x) for x in os.environ.get('CLASSES', '17,6,3').split(',')]
GAIN = float(os.environ.get('GAIN', '0.8')); S = float(os.environ.get('STRENGTH', '0.75'))
VEH_MARGIN = int(os.environ.get('VEH_MARGIN', '0')); CMIN = float(os.environ.get('CMIN', '0'))
LUMA = float(os.environ.get('LUMA', '0')); LUMA_SIG = float(os.environ.get('LUMA_SIG', '8'))
VEH = [52, 54, 55, 56, 57, 58, 59, 60, 61, 62]


def process(im, car, lab):
    m = np.isin(lab, CLASSES).astype(np.float32)
    if m.sum() < 500: return im
    A = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); C = cv2.cvtColor(car, cv2.COLOR_BGR2LAB).astype(np.float32)
    if VEH_MARGIN > 0:
        r = max(1, round(VEH_MARGIN * im.shape[1] / 1920))
        near = cv2.dilate(np.isin(lab, VEH).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        m = m * (1 - near.astype(np.float32))
    if LUMA > 0:
        wl = cv2.GaussianBlur(m, (0, 0), 3.0)
        sg = LUMA_SIG * im.shape[1] / 1920
        d = cv2.GaussianBlur(C[..., 0], (0, 0), sg) - cv2.GaussianBlur(A[..., 0], (0, 0), sg)
        A[..., 0] = A[..., 0] + LUMA * wl * d
    w = cv2.GaussianBlur(m, (0, 0), 1.5)
    if CMIN > 0:
        ch = np.hypot(C[..., 1] - 128, C[..., 2] - 128)
        w = w * np.clip((ch - CMIN) / CMIN, 0, 1)
    w = (w * S)[..., None]
    tgt = 128 + (C[..., 1:] - 128) * GAIN
    A[..., 1:] = A[..., 1:] * (1 - w) + tgt * w
    return cv2.cvtColor(np.clip(np.round(A), 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def load(rgbs, labs, i, W, H):
    car = cv2.resize(np.ascontiguousarray(cv2.imread(rgbs[i])[:, :, ::-1]), (W, H), interpolation=cv2.INTER_AREA)
    l = np.array(Image.open(labs[i])); l = l if l.ndim == 2 else l[..., 0]
    return car, cv2.resize(l, (W, H), interpolation=cv2.INTER_NEAREST)


if __name__ == '__main__':
    a = sys.argv[1:]
    frame = None
    if a[0] == '--frame': frame = int(a[1]); a = a[2:]
    IN, RGB, LABD, OUT = a[:4]
    rgbs = sorted(glob.glob(os.path.join(RGB, '*.png'))); labs = sorted(glob.glob(os.path.join(LABD, '*.png')))
    cap = cv2.VideoCapture(IN); W, H = int(cap.get(3)), int(cap.get(4))
    if frame is not None:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame); f = cap.read()[1]
        cv2.imwrite(OUT, process(f, *load(rgbs, labs, frame, W, H))); sys.exit()
    vw = cv2.VideoWriter(OUT, fourcc_for(OUT), cap.get(5) or 30, (W, H)); n = 0
    while True:
        ok, f = cap.read()
        if not ok: break
        if n < min(len(rgbs), len(labs)): f = process(f, *load(rgbs, labs, n, W, H))
        vw.write(f); n += 1
    vw.release(); print(f'wrote {OUT}: {n} frames (scene chroma from CARLA, classes {CLASSES}, gain {GAIN}, strength {S})')
