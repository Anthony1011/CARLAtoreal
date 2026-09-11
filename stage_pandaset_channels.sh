#!/bin/bash
# Generate every conditioning channel for the 8,240 PandaSet frames, then link them into the corpus.
#
# STAGED SEPARATELY on purpose. The corpus already holds 23,406 Mapillary and Cityscapes pairs whose
# channels were generated months ago and are symlinked through. Running the generators over the
# combined directory would recompute all of them -- hours of GPU for output that already exists, and
# it would write real files over symlinks. So PandaSet gets its own directory, and only the finished
# maps are linked in.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n $CARLA2REAL_ENV"
STAGE=$CARLA2REAL_DATA/pandaset_stage
DST=$CARLA2REAL_DATA/training_pandaset
LOG=$BASE/pix2pixHD/checkpoints/pandaset_stage_log.txt
DIRS="train_label train_edge train_depth train_normal train_chroma train_texture"

echo "=== staging PandaSet channels $(date) ===" > "$LOG"
N=$(ls "$STAGE/train_img" | wc -l)
echo "  $N source frames" >> "$LOG"
[ "$N" -lt 8000 ] && { echo "  ABORT: flatten did not run" >> "$LOG"; exit 1; }

# RE-RUNNABLE. Each stage is skipped when its output is already complete, so fixing one broken
# call does not mean regenerating the two hours of Mask2Former labels that already succeeded.
done_already () { [ "$(ls "$STAGE/$1" 2>/dev/null | wc -l)" -eq "$N" ]; }

echo "--- labels (Mask2Former) $(date)" >> "$LOG"
if done_already train_label; then echo "  already complete, skipped" >> "$LOG"; else
$CE python3 -u $BASE/gen_mapillary_labels_paths.py --src_dir "$STAGE/train_img" \
    --out_dir "$STAGE/train_label" >> "$LOG" 2>&1
fi
L=$(ls "$STAGE/train_label" 2>/dev/null | wc -l)
echo "  labels: $L / $N" >> "$LOG"
[ "$L" -eq "$N" ] || { echo "  ABORT: label generation incomplete" >> "$LOG"; exit 1; }

echo "--- edges $(date)" >> "$LOG"
# three arguments: rgb, LABEL, out. Passing two raised IndexError and produced nothing.
if done_already train_edge; then echo "  already complete, skipped" >> "$LOG"; else
$CE python3 -u $BASE/gen_edges_denoised.py "$STAGE/train_img" "$STAGE/train_label" "$STAGE/train_edge" \
    >> "$LOG" 2>&1
fi
echo "--- depth + normals (MoGe) $(date)" >> "$LOG"
if done_already train_depth; then echo "  already complete, skipped" >> "$LOG"; else
$CE python3 -u $BASE/gen_depth_moge.py "$STAGE/train_label" "$STAGE/train_img" \
    "$STAGE/train_depth" "$STAGE/train_normal" >> "$LOG" 2>&1
fi
echo "--- chroma $(date)" >> "$LOG"
if done_already train_chroma; then echo "  already complete, skipped" >> "$LOG"; else
$CE python3 -u $BASE/gen_chroma_maps.py --src_dir "$STAGE/train_img" --label_dir "$STAGE/train_label" \
    --out_dir "$STAGE/train_chroma" >> "$LOG" 2>&1
fi
echo "--- roughness $(date)" >> "$LOG"
if done_already train_texture; then echo "  already complete, skipped" >> "$LOG"; else
$CE python3 -u $BASE/gen_texture_energy.py "$STAGE/train_img" "$STAGE/train_texture" >> "$LOG" 2>&1
fi

echo "--- gate before linking $(date)" >> "$LOG"
bad=0
for d in $DIRS; do
  M=$(ls "$STAGE/$d" 2>/dev/null | wc -l)
  echo "  $d: $M / $N" >> "$LOG"
  [ "$M" -eq "$N" ] || bad=1
done
[ "$bad" = 0 ] || { echo "  ABORT: a channel is short; linking a partial set would misalign every pair after the gap" >> "$LOG"; exit 1; }

echo "--- link into the corpus $(date)" >> "$LOG"
for d in train_img $DIRS; do
  for f in "$STAGE/$d"/pshr_*; do
    [ -e "$f" ] || continue
    ln -sf "$f" "$DST/$d/$(basename "$f")"
  done
  echo "  $d: $(ls "$DST/$d" | wc -l) total in corpus" >> "$LOG"
done

# the check that was missing last time: the corpus must actually CONTAIN PandaSet, not merely be
# self-consistent. 23,406 consistent entries passed the old gate while holding no new data at all.
P=$(ls "$DST/train_img" | grep -c '^pshr_')
T=$(ls "$DST/train_img" | wc -l)
echo "  corpus: $T pairs, $P of them PandaSet" >> "$LOG"
[ "$P" -ge 8000 ] && echo "=== CORPUS READY $(date) ===" >> "$LOG" \
                  || { echo "=== PANDASET MISSING FROM CORPUS $(date) ===" >> "$LOG"; exit 1; }
