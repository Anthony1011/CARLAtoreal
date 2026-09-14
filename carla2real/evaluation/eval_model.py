#!/usr/bin/env python3
"""Score one trained model on every axis this project judges by, and emit a JSON row.

Exists so an unattended run can COMPARE experiments rather than just produce them. Every number
here has a history of being misread, so each is defined to avoid the specific trap:

  veg_near / veg_far    vegetation detail against real photographs, reported separately because
                        raising near detail while distant trees stay bad is not a fix (v63).
  road_ratio            high-pass energy on ROAD-labelled pixels against the parent. The gate.
                        Above 1.5 the model is inventing grain on a surface the label calls flat.
  warped_resid          instability with frame t-1 warped onto t by optical flow, so detail that
                        merely MOVED cancels. Plain alternation counts a sharp stationary surface
                        sweeping past the camera as flicker and has already reversed one decision.
  detail                high-pass energy overall. Reported next to road_ratio on purpose: detail
                        rising while road_ratio also rises is a crosshatch, not an improvement.
  tone_car / tone_road  mean luminance by class against the parent. Fine-tuning this parent on this
                        corpus drifts tone -- cars darker, road brighter -- and it is invisible to
                        every other metric here.

  usage: eval_model.py <model_name> <parent_name> <phase> [--out results.json]
"""
import glob
import json
import os
import sys

from carla2real.config import DATA, ROOT, OUT  # noqa: E402

import cv2
import numpy as np

R = os.path.join(ROOT, 'pix2pixHD/results')
LBL = os.path.join(DATA, 'training_v12_mapillary')
MODEL, PARENT, PHASE = sys.argv[1], sys.argv[2], sys.argv[3]
OUT = sys.argv[sys.argv.index('--out') + 1] if '--out' in sys.argv else None

lab_dir = os.path.join(LBL, PHASE.replace('_latest', '') + '_label')
labs = sorted(glob.glob(lab_dir + '/*.png'))
if not labs:
    raise SystemExit(f'no labels at {lab_dir}')


def frames(name):
    return sorted(glob.glob(f'{R}/{name}/{PHASE}/images/*_synthesized_image.jpg'))


def hp(g, s=2.0):
    return np.abs(g - cv2.GaussianBlur(g, (0, 0), s))


def measure(fs, idxs):
    """Per-class high-pass energy and luminance, plus flow-warped instability."""
    acc = {k: [] for k in ('road', 'veg', 'build', 'car')}
    lum = {k: [] for k in ('road', 'veg', 'build', 'car')}
    ids = {'road': (13,), 'veg': (30,), 'build': (17,), 'car': (55, 61)}
    detail, resid = [], []
    prev_g = prev_raw = None
    fb = dict(pyr_scale=0.5, levels=3, winsize=21, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
    for i in idxs:
        if i >= len(fs) or i >= len(labs):
            continue
        img = cv2.imread(fs[i])
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h = hp(g)
        detail.append(float(h.mean()))
        lm = cv2.imread(labs[i], 0)
        lm = cv2.resize(lm, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        for k, cid in ids.items():
            m = cv2.erode(np.isin(lm, cid).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
            if m.sum() > 5000:
                acc[k].append(float(h[m].mean()))
                lum[k].append(float(g[m].mean()))
        # instability: warp the previous frame forward, so movement cancels and only
        # reinvented texture is left
        if prev_g is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_raw, g.astype(np.uint8), None, **fb)
            hh, ww = g.shape
            gx, gy = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))
            warped = cv2.remap(prev_g, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR)
            resid.append(float(np.abs(hp(g) - warped).mean()))
        prev_g = hp(g)
        prev_raw = g.astype(np.uint8)
    out = {f'{k}_tex': (float(np.mean(v)) if v else None) for k, v in acc.items()}
    out.update({f'{k}_lum': (float(np.mean(v)) if v else None) for k, v in lum.items()})
    out['detail'] = float(np.mean(detail)) if detail else None
    out['warped_resid'] = float(np.mean(resid)) if resid else None
    return out


def veg_by_distance(fs, idxs):
    """Vegetation detail split near/far by connected-region area -- a distant tree is few pixels.

    Depth is a monocular estimate and not reliable enough to bucket on, which is why area is used.
    """
    near, far = [], []
    for i in idxs:
        if i >= len(fs) or i >= len(labs):
            continue
        img = cv2.imread(fs[i])
        if img is None:
            continue
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        h = hp(g, 1.6)
        lm = cv2.imread(labs[i], 0)
        lm = cv2.resize(lm, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        m = (lm == 30).astype(np.uint8)
        if m.sum() < 2000:
            continue
        n, lab_im, stats, _ = cv2.connectedComponentsWithStats(m, 8)
        for j in range(1, n):
            area = stats[j, cv2.CC_STAT_AREA]
            if area < 200:
                continue
            reg = (lab_im == j)
            v = float(h[reg].mean()) * 100.0
            (near if area > 8000 else far).append(v)
    return (float(np.mean(near)) if near else None), (float(np.mean(far)) if far else None)


idxs = list(range(200, 320, 4))
fm, fp = frames(MODEL), frames(PARENT)
if len(fm) < 100:
    raise SystemExit(f'{MODEL}: only {len(fm)} frames rendered')

row = {'model': MODEL, 'parent': PARENT, 'phase': PHASE, 'frames': len(fm)}
m, p = measure(fm, idxs), measure(fp, idxs) if len(fp) >= 100 else None
row.update(m)
row['veg_near'], row['veg_far'] = veg_by_distance(fm, idxs)
if row['veg_near'] and row['veg_far']:
    row['veg_far_over_near'] = row['veg_far'] / row['veg_near']
if p:
    row['road_ratio'] = m['road_tex'] / p['road_tex'] if p['road_tex'] else None
    for k in ('car', 'road', 'build', 'veg'):
        if m[f'{k}_lum'] and p[f'{k}_lum']:
            row[f'tone_{k}'] = m[f'{k}_lum'] - p[f'{k}_lum']
    row['detail_ratio'] = m['detail'] / p['detail'] if p['detail'] else None
    row['resid_ratio'] = (m['warped_resid'] / p['warped_resid']
                          if m['warped_resid'] and p['warped_resid'] else None)

print(json.dumps(row, indent=2))
if OUT:
    rows = []
    if os.path.exists(OUT):
        try:
            rows = json.load(open(OUT))
        except Exception:
            rows = []
    rows = [r for r in rows if not (r.get('model') == MODEL and r.get('phase') == PHASE)]
    rows.append(row)
    json.dump(rows, open(OUT, 'w'), indent=2)
    print(f'  appended to {OUT} ({len(rows)} rows)')
