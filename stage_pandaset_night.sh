#!/bin/bash
# Build the night corpus with PandaSet in place of the four unlicensed night videos.
#
# WHAT IS SWAPPED. The night corpus is 6,826 pairs:
#     ngt_           3,600  videos 10-13, no licence          <- replaced
#     dz_            2,670  Dark Zurich, licensed for research
#     v...             534
#     mnight_mvhr_      22  Mapillary, night subset
# 4,320 PandaSet night frames (psngt_) go in place of ngt_, giving 7,546 pairs. Everything else is
# symlinked through, so the data is again the only variable.
#
# NIGHT HAS NO CHROMA AND DOES HAVE LIGHT. The night architecture is --light_input, not
# --chroma_input: a colour prior is close to useless when the scene is lit by sodium lamps, while an
# emissive map tells the generator where the light sources are. Building the wrong channel set here
# would leave the loader one channel short, and load_network would drop the first conv to random
# init without raising.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n $CARLA2REAL_ENV"
STAGE=$CARLA2REAL_DATA/pandaset_night_stage
OLD=$CARLA2REAL_DATA/training_v51_night
DST=$CARLA2REAL_DATA/training_pandaset_night
LOG=$BASE/pix2pixHD/checkpoints/pandaset_night_log.txt
DIRS="train_label train_edge train_depth train_normal train_light"

echo "=== PandaSet night corpus $(date) ===" > "$LOG"
N=$(ls "$STAGE/train_img" 2>/dev/null | wc -l)
echo "  $N PandaSet night frames staged" >> "$LOG"
[ "$N" -lt 3600 ] && { echo "  ABORT: fewer than the 3,600 being replaced" >> "$LOG"; exit 1; }

for d in train_img $DIRS; do mkdir -p "$DST/$d"; done

# [1] carry the licensed half through, dropping only ngt_
echo "--- [1] link the non-video half $(date)" >> "$LOG"
for d in train_img $DIRS; do
  n=0
  for f in "$OLD/$d"/*; do
    b=$(basename "$f")
    case "$b" in ngt_*) continue ;; esac
    [ -e "$f" ] || continue
    ln -sf "$f" "$DST/$d/$b"; n=$((n+1))
  done
  echo "  $d: $n linked (ngt_ excluded)" >> "$LOG"
done

# [2] channels for the PandaSet night frames
done_already () { [ "$(ls "$STAGE/$1" 2>/dev/null | wc -l)" -eq "$N" ]; }

echo "--- [2] labels $(date)" >> "$LOG"
if done_already train_label; then echo "  skipped" >> "$LOG"; else
$CE python3 -u $BASE/gen_mapillary_labels_paths.py --src_dir "$STAGE/train_img" \
    --out_dir "$STAGE/train_label" >> "$LOG" 2>&1
fi
L=$(ls "$STAGE/train_label" 2>/dev/null | wc -l)
[ "$L" -eq "$N" ] || { echo "  ABORT: labels $L of $N" >> "$LOG"; exit 1; }

echo "--- [3] edges $(date)" >> "$LOG"
if done_already train_edge; then echo "  skipped" >> "$LOG"; else
$CE python3 -u $BASE/gen_edges_denoised.py "$STAGE/train_img" "$STAGE/train_label" "$STAGE/train_edge" \
    >> "$LOG" 2>&1
fi
echo "--- [4] depth + normals $(date)" >> "$LOG"
if done_already train_depth; then echo "  skipped" >> "$LOG"; else
$CE python3 -u $BASE/gen_depth_moge.py "$STAGE/train_label" "$STAGE/train_img" \
    "$STAGE/train_depth" "$STAGE/train_normal" >> "$LOG" 2>&1
fi
echo "--- [5] light maps $(date)" >> "$LOG"
if done_already train_light; then echo "  skipped" >> "$LOG"; else
$CE python3 -u $BASE/gen_light_maps.py --src_dir "$STAGE/train_img" --out_dir "$STAGE/train_light" \
    >> "$LOG" 2>&1
fi

echo "--- [6] gate before linking $(date)" >> "$LOG"
bad=0
for d in $DIRS; do
  M=$(ls "$STAGE/$d" 2>/dev/null | wc -l)
  echo "  $d: $M / $N" >> "$LOG"
  [ "$M" -eq "$N" ] || bad=1
done
[ "$bad" = 0 ] || { echo "  ABORT: a channel is short" >> "$LOG"; exit 1; }

echo "--- [7] link into the corpus $(date)" >> "$LOG"
for d in train_img $DIRS; do
  for f in "$STAGE/$d"/psngt_*; do
    [ -e "$f" ] || continue
    ln -sf "$f" "$DST/$d/$(basename "$f")"
  done
done

# presence, not just consistency -- the sunny build produced a perfectly consistent corpus
# containing none of the new data, and every count check passed
P=$(ls "$DST/train_img" | grep -c '^psngt_')
T=$(ls "$DST/train_img" | wc -l)
echo "  corpus: $T pairs, $P of them PandaSet night" >> "$LOG"
[ "$P" -ge 3600 ] && echo "=== NIGHT CORPUS READY $(date) ===" >> "$LOG" \
                  || { echo "=== PANDASET NIGHT MISSING $(date) ===" >> "$LOG"; exit 1; }
