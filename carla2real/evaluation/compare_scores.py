#!/usr/bin/env python3
"""Vision Pilot scores for every delivered tag, against the standing baselines.

Reads the per-town score JSONs render_model.sh writes into CARLA/logs_<tag>/ and prints one row
per tag averaged over the towns it shares with its baseline. Averaging over a tag's own towns
would compare a two-town model against a five-town baseline on different content, which is not a
comparison at all -- so every row here is restricted to the intersection.

  usage: compare_scores.py [tag ...]      (default: the PandaSet arm plus its baselines)
"""
import json
import os
import sys

from carla2real.config import DATA, ROOT, OUT  # noqa: E402

DEST = OUT
SUNNY_BASE, NIGHT_BASE = 'v50m', 'v59'
DEFAULT = ['v50m', 'v68', 'v70', 'v71', 'v72', 'v59', 'v69']


def load(tag):
    """town -> flat metrics, for one delivered tag."""
    d, out = os.path.join(DEST, f'logs_{tag}'), {}
    if not os.path.isdir(d):
        return out
    for f in sorted(os.listdir(d)):
        if not f.endswith('_score.json'):
            continue
        try:
            r = json.load(open(os.path.join(d, f)))
        except Exception:
            continue
        r = r[0] if isinstance(r, list) and r else r
        if not isinstance(r, dict):
            continue
        out[f[:-len('_score.json')]] = {
            'cipo': r.get('cipo', {}).get('recall'),
            'fa': r.get('cipo', {}).get('false_alarm'),
            'rng': r.get('rng', {}).get('mae'),
            'lane': r.get('lane', {}).get('mae'),
            'cov': r.get('lane', {}).get('coverage'),
            'jit': r.get('stability', {}).get('cte_jitter_p99'),
        }
    return out


def mean(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return sum(vals) / len(vals) if vals else None


def row(tag, towns, sc):
    m = {k: mean([sc[t][k] for t in towns if t in sc]) for k in
         ('cipo', 'fa', 'rng', 'lane', 'cov', 'jit')}
    m['tag'], m['n'] = tag, len(towns)
    return m


def fmt(v, p=3):
    return f'{v:.{p}f}' if isinstance(v, (int, float)) else '  --  '


tags = sys.argv[1:] or DEFAULT
scores = {t: load(t) for t in tags}

for weather, base in (('sunny', SUNNY_BASE), ('night', NIGHT_BASE)):
    group = [t for t in tags if any(k.endswith(weather) for k in scores.get(t, {}))]
    if base not in group:
        continue
    print(f'\n=== {weather}  (baseline {base}) ' + '=' * 34)
    print(f'{"tag":<8}{"towns":>6}{"CIPO":>8}{"falsealm":>10}{"rangeMAE":>10}'
          f'{"laneMAE":>9}{"lanecov":>9}{"jitter":>8}   vs baseline')
    bt = {t for t in scores[base] if t.endswith(weather)}
    brow = row(base, sorted(bt), scores[base])
    for tag in group:
        st = {t for t in scores[tag] if t.endswith(weather)}
        shared = sorted(bt & st)
        if not shared:
            continue
        # the baseline is re-averaged over the SAME towns, so the delta is not a content artefact
        r, b = row(tag, shared, scores[tag]), row(base, shared, scores[base])
        d = ''
        if tag != base and r['cipo'] is not None and b['cipo'] is not None:
            dc = (r['cipo'] - b['cipo']) * 100
            dl = (r['lane'] - b['lane']) if r['lane'] and b['lane'] else 0
            d = f'CIPO {dc:+.1f} pts, lane {dl:+.3f} m'
        print(f'{tag:<8}{len(shared):>6}{fmt(r["cipo"]):>8}{fmt(r["fa"]):>10}'
              f'{fmt(r["rng"]):>10}{fmt(r["lane"]):>9}{fmt(r["cov"]):>9}'
              f'{fmt(r["jit"]):>8}   {d}')
    miss = [t for t in group if not ({x for x in scores[t] if x.endswith(weather)} & bt)]
    if miss:
        print(f'  (no shared towns yet: {", ".join(miss)})')
    print(f'  baseline {base} over all its towns: CIPO {fmt(brow["cipo"])}  '
          f'lane {fmt(brow["lane"])}  ({brow["n"]} towns)')
