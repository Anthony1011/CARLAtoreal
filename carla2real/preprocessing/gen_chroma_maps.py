"""Chroma prior: tell the generator what COLOUR an object is, instead of letting it guess.

THE FAILURE THIS TARGETS. Mapillary class 55 means "car". It says nothing about whether this
particular car is silver or red. The generator therefore invents a colour -- and being a
per-frame image model with no memory, it invents a slightly different one every frame. Visible
result: red splotches wandering across a silver SUV, which is the "some parts painted red some
isn't" complaint, and part of the flicker as well.

Instance boundaries were tried for this (v48) and measured WORSE than doing nothing: car hue
spread 80.1 vs soup0.8's 65.2 on identical frames. Boundaries tell the model where an object
ENDS. They never told it what the object IS. This channel does.

TRAIN/INFERENCE CONTRACT -- the same one that made the edge channel and the light channel work,
and the only pattern in this project with a track record:
    training   computed from the REAL PHOTOGRAPH, so the target colour is the true one
    inference  computed from CARLA's render, so CARLA dictates the colour
The model learns "paint this region the colour the prior says", and at inference CARLA says it.

WHY IT SHOULD ALSO CUT FLICKER, for free: CARLA's car colour is identical in every frame. A
model conditioned on it has no reason to drift frame to frame. That is temporal consistency
obtained from a per-frame model without any temporal machinery.

DESIGN, and why each choice matters:
  - CHROMA ONLY, no luminance. Converted to Lab, L is pinned flat and only a/b survive. Passing
    brightness would leak CARLA's shading and lighting into the prior and the output would drift
    back toward looking like CARLA -- the exact thing this pipeline exists to avoid.
  - MASK-AWARE, LIGHT BLUR. The per-region median fill already guarantees a flat hint rather
    than a texture copy, so heavy blurring adds nothing and actively hurts: blurring across the
    neutral background dilutes each car's colour back toward grey, which a first attempt at
    sigma 25 did visibly. The blur is therefore small AND normalised by the mask, so colour is
    smoothed WITHIN objects and feathered at their edges without being averaged into the grey.
  - THING CLASSES ONLY. Cars, trucks, buses, bikes, riders, people. Road, sky and vegetation are
    left neutral grey -- their colour is not in dispute and constraining them would only fight
    the model. Neutral grey (128) is the a/b zero point, i.e. "no opinion".
  - PER-INSTANCE FILL, not a blur of the raw image. Each connected region is filled with its own
    MEDIAN colour before blurring, so a car reads as one flat colour rather than a smear of
    itself and the road behind it. Median, not mean, so a bright specular highlight or a dark
    shadow does not drag the whole car's prior.

STUFF CHROMA (--stuff, added 11-03 for v98cc). User on v98scvs: "muddy, the Town10 banners' colours are all
muddy, dystopian". Cause: buildings/banners got the neutral "no opinion" prior, so the model paints them a safe
overcast grey. With --stuff, facade-type classes (building, wall, fence, bridge, banner, billboard) get a LOCAL hint:
the photo's own a,b, mask-aware blurred (sigma --stuff_sigma px per 1024 of width) so it is a colour field, not
texture, and gated by chroma (ramps in over --stuff_cmin..2x) so plain grey/beige walls stay "no opinion" and only
genuinely colourful patches (banners, painted shopfronts) carry a hint. Same contract: real photo at train time,
CARLA at inference (with --gain <1 there, as CARLA is more saturated than real photos).

Usage:
  gen_chroma_maps.py --src_dir <images> --label_dir <labels> --out_dir <out> [--sigma 25]
  --labels mapillary (default, ids 54-58,19-22) or carla (ids 14-19,12,13)
"""
import argparse, glob, os
import cv2
import numpy as np
from PIL import Image
from scipy import ndimage

# Mapillary-65 thing classes: person/rider variants and the vehicle block
MAPILLARY_THINGS = [19, 20, 21, 22, 54, 55, 56, 57, 58, 61]
# CARLA 0.9.16 tags: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle
CARLA_THINGS = [12, 13, 14, 15, 16, 17, 18, 19]
# Cityscapes 19-class trainIds: person, rider, car, truck, bus, train, motorcycle, bicycle.
# The corpus is MIXED -- cshr_* files are Cityscapes-space, mvhr_/nrhr_* are Mapillary-65 --
# and the two spaces collide: id 13 is CAR in Cityscapes but ROAD in Mapillary. Unioning the
# lists would paint every road with a car's colour on 92% of the corpus, so the space is
# detected per frame instead (max id > 30 can only be Mapillary; Cityscapes tops out at 18).
CITYSCAPES_THINGS = [11, 12, 13, 14, 15, 16, 17, 18]

ap = argparse.ArgumentParser()
ap.add_argument('--src_dir', required=True, help='RGB: real photos at train time, CARLA render at inference')
ap.add_argument('--label_dir', required=True)
ap.add_argument('--out_dir', required=True)
ap.add_argument('--sigma', type=float, default=8.0, help='edge feather; small -- the median fill is already flat')
ap.add_argument('--gain', type=float, default=1.0, help='chroma amplification; >1 exaggerates colour deviation')
ap.add_argument('--labels', choices=['mapillary', 'carla', 'auto'], default='auto')
ap.add_argument('--min_px', type=int, default=250)
ap.add_argument('--stuff', action='store_true', help='also give facade classes a local colour hint (see STUFF CHROMA)')
ap.add_argument('--stuff_classes', default='3,6,16,17,32,35')
ap.add_argument('--stuff_sigma', type=float, default=4.0)
ap.add_argument('--stuff_cmin', type=float, default=12.0)
ap.add_argument('--stuff_gain', type=float, default=1.0)
ap.add_argument('--carla_bgr', action='store_true',
                help='CARLA PNGs are BGR-saved-as-RGB; set when --src_dir is a CARLA capture')
a = ap.parse_args()

FIXED = {'mapillary': MAPILLARY_THINGS, 'carla': CARLA_THINGS}.get(a.labels)
os.makedirs(a.out_dir, exist_ok=True)

srcs = sorted(glob.glob(os.path.join(a.src_dir, '*.png')) + glob.glob(os.path.join(a.src_dir, '*.jpg')))
labs = sorted(glob.glob(os.path.join(a.label_dir, '*.png')))
n = min(len(srcs), len(labs))
if n == 0:
    raise SystemExit('no inputs: %d src, %d label' % (len(srcs), len(labs)))
if len(srcs) != len(labs):
    print('  warning: %d src vs %d labels, using first %d' % (len(srcs), len(labs), n))

covered = []
for i in range(n):
    img = cv2.imread(srcs[i], cv2.IMREAD_COLOR)
    if img is None:
        continue
    if a.carla_bgr:
        img = img[:, :, ::-1].copy()
    L = np.array(Image.open(labs[i]))
    if L.ndim == 3:
        L = L[:, :, 2] if a.labels == 'carla' else L[:, :, 0]
    if L.shape[:2] != img.shape[:2]:
        L = cv2.resize(L, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    out = np.full_like(lab, 0)
    out[:, :, 0] = 128            # flat L: this channel carries no brightness, by design
    out[:, :, 1] = 128            # neutral a
    out[:, :, 2] = 128            # neutral b

    things = FIXED if FIXED is not None else (
        MAPILLARY_THINGS if L.max() > 30 else CITYSCAPES_THINGS)
    m_all = np.isin(L, things)
    if m_all.sum() >= a.min_px:
        cc, nc = ndimage.label(m_all)
        for j in range(1, nc + 1):
            r = (cc == j)
            if r.sum() < a.min_px:
                continue
            # median so a highlight or shadow cannot drag the region's colour
            out[:, :, 1][r] = np.median(lab[:, :, 1][r])
            out[:, :, 2][r] = np.median(lab[:, :, 2][r])
    covered.append(float(m_all.mean()) * 100)

    if a.gain != 1.0:
        for ch in (1, 2):
            out[:, :, ch] = np.clip(128 + (out[:, :, ch].astype(np.float32) - 128) * a.gain, 0, 255)
    # mask-aware blur: smooth the colour field and the mask together, then divide, so an
    # object's colour feathers at its own boundary instead of being averaged into the grey
    if a.sigma > 0 and m_all.sum() >= a.min_px:
        w = cv2.GaussianBlur(m_all.astype(np.float32), (0, 0), a.sigma)
        base = np.full_like(out, 128)
        for ch in (1, 2):
            f = out[:, :, ch].astype(np.float32) * m_all
            num = cv2.GaussianBlur(f, (0, 0), a.sigma)
            sm = np.where(w > 1e-3, num / np.maximum(w, 1e-3), 128.0)
            # blend toward neutral where the mask is weak, so there is no hard cut at the edge
            base[:, :, ch] = np.clip(sm * np.clip(w, 0, 1) + 128.0 * (1 - np.clip(w, 0, 1)), 0, 255)
        out = base.astype(np.uint8)
    if a.stuff and L.max() > 30:   # Mapillary-space labels only
        sm_ = np.isin(L, [int(x) for x in a.stuff_classes.split(',')]) & ~m_all
        if sm_.sum() >= a.min_px:
            sg = a.stuff_sigma * img.shape[1] / 1024.0
            mf = sm_.astype(np.float32); wsum = cv2.GaussianBlur(mf, (0, 0), sg)
            ab = lab[:, :, 1:].astype(np.float32) - 128
            loc = np.stack([cv2.GaussianBlur(ab[..., k] * mf, (0, 0), sg) for k in (0, 1)], -1) / np.maximum(wsum, 1e-3)[..., None]
            ch = np.hypot(loc[..., 0], loc[..., 1])
            g = np.clip((ch - a.stuff_cmin) / a.stuff_cmin, 0, 1) * sm_
            g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 1.0)
            for k in (0, 1):
                cur = out[:, :, k + 1].astype(np.float32)
                out[:, :, k + 1] = np.clip(cur * (1 - g) + (128 + loc[..., k] * a.stuff_gain) * g, 0, 255).astype(np.uint8)
    rgbish = cv2.cvtColor(out, cv2.COLOR_LAB2BGR)
    Image.fromarray(cv2.cvtColor(rgbish, cv2.COLOR_BGR2RGB)).save(
        os.path.join(a.out_dir, os.path.splitext(os.path.basename(labs[i]))[0] + '.png'))
    if (i + 1) % 2000 == 0:
        print('  %d/%d' % (i + 1, n), flush=True)

zero = sum(1 for c in covered if c < 0.01)
print('  wrote %d chroma maps -> %s   thing coverage %.2f%% of pixels   sigma %.0f'
      % (n, a.out_dir, float(np.mean(covered)) if covered else 0.0, a.sigma))
print('  %d frames (%.1f%%) contain no thing classes at all and are neutral by design'
      % (zero, 100.0 * zero / max(n, 1)))
