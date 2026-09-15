#!/bin/bash
# Generate every conditioning channel for the flattened ZOD frames, then link them into the two
# corpora. Same shape as stage_pandaset_channels.sh, and re-runnable for the same reason: each
# stage is skipped when its output is already complete, so fixing one broken generator does not
# mean regenerating the hours of Mask2Former labels that already succeeded.
#
# The gate at the end checks PRESENCE, not just consistency. A corpus of 31,646 entries that agreed
# perfectly across every channel and contained zero new frames passed the old count check, and four
# training runs were attributed to data they never saw.
set -u
. "$(dirname "${BASH_SOURCE[0]}")/../../configs/config.sh"
BASE=$CARLA2REAL_ROOT
CE="conda run -n $CARLA2REAL_ENV"
STAGE=$CARLA2REAL_DATA/zod_stage
LOG=$BASE/pix2pixHD/checkpoints/zod_stage_log.txt
REF=$CARLA2REAL_DATA/training_v49_chroma/train_texture

echo "=== staging ZOD channels $(date) ===" > "$LOG"

stage_one () {         # stage_one <sunny|night> <dst corpus> <prefix>
  local W=$1 DST=$2 PFX=$3 S=$STAGE/$1
  local N; N=$(ls "$S/train_img" 2>/dev/null | wc -l)
  echo "--- $W: $N source frames $(date)" >> "$LOG"
  [ "$N" -lt 1000 ] && { echo "  ABORT: flatten did not produce enough for $W" >> "$LOG"; return 1; }

  local DIRS="train_label train_edge train_depth train_normal"
  if [ "$W" = night ]; then DIRS="$DIRS train_light"
  else DIRS="$DIRS train_chroma train_texture"; fi

  done_already () { [ "$(ls "$S/$1" 2>/dev/null | wc -l)" -eq "$N" ]; }

  echo "  labels (Mask2Former)" >> "$LOG"
  done_already train_label || $CE python3 -u $BASE/gen_mapillary_labels_paths.py \
      --src_dir "$S/train_img" --out_dir "$S/train_label" >> "$LOG" 2>&1
  local L; L=$(ls "$S/train_label" 2>/dev/null | wc -l)
  [ "$L" -eq "$N" ] || { echo "  ABORT $W: labels $L/$N" >> "$LOG"; return 1; }

  # three arguments: rgb, LABEL, out. Passing two raises IndexError and produces nothing.
  echo "  edges" >> "$LOG"
  done_already train_edge || $CE python3 -u $BASE/gen_edges_denoised.py \
      "$S/train_img" "$S/train_label" "$S/train_edge" >> "$LOG" 2>&1

  echo "  depth + normals (MoGe)" >> "$LOG"
  done_already train_depth || $CE python3 -u $BASE/gen_depth_moge.py \
      "$S/train_label" "$S/train_img" "$S/train_depth" "$S/train_normal" >> "$LOG" 2>&1

  if [ "$W" = night ]; then
    echo "  light maps" >> "$LOG"
    done_already train_light || $CE python3 -u $BASE/gen_light_maps.py \
        --src_dir "$S/train_img" --out_dir "$S/train_light" >> "$LOG" 2>&1
  else
    echo "  chroma" >> "$LOG"
    done_already train_chroma || $CE python3 -u $BASE/gen_chroma_maps.py \
        --src_dir "$S/train_img" --label_dir "$S/train_label" --out_dir "$S/train_chroma" >> "$LOG" 2>&1
    # no --match-ref: these are real photographs, so they ARE the reference distribution. Matching
    # is only needed for CARLA renders at inference, whose roughness sits near twice this median.
    echo "  roughness" >> "$LOG"
    done_already train_texture || $CE python3 -u -m carla2real.preprocessing.gen_texture_energy \
        "$S/train_img" "$S/train_texture" >> "$LOG" 2>&1
  fi

  echo "  gate before linking" >> "$LOG"
  local bad=0 M
  for d in $DIRS; do
    M=$(ls "$S/$d" 2>/dev/null | wc -l)
    echo "    $d: $M / $N" >> "$LOG"
    [ "$M" -eq "$N" ] || bad=1
  done
  [ "$bad" = 0 ] || { echo "  ABORT $W: a channel is short; linking a partial set misaligns every pair after the gap" >> "$LOG"; return 1; }

  echo "  linking into $DST" >> "$LOG"
  for d in train_img $DIRS; do
    mkdir -p "$DST/$d"
    for f in "$S/$d"/${PFX}*; do
      [ -e "$f" ] || continue
      ln -sf "$(readlink -f "$f")" "$DST/$d/$(basename "$f")"
    done
  done
  local P T
  P=$(ls "$DST/train_img" | grep -c "^$PFX") ; T=$(ls "$DST/train_img" | wc -l)
  echo "  $DST: $T pairs, $P of them ZOD" >> "$LOG"
  [ "$P" -ge 1000 ] && echo "=== $W CORPUS READY ($T pairs) $(date) ===" >> "$LOG" \
                    || { echo "=== ZOD MISSING FROM $W CORPUS $(date) ===" >> "$LOG"; return 1; }
}

# The ZOD corpora are built ON TOP of the PandaSet ones -- both licensed sources together, which is
# the whole point of using both. Copy the symlink farm rather than mutating it, so training_pandaset
# stays intact as its own comparison point.
for pair in "sunny training_pandaset training_pz" "night training_pandaset_night training_pz_night"; do
  set -- $pair
  SRC=$CARLA2REAL_DATA/$2; NEW=$CARLA2REAL_DATA/$3
  if [ ! -d "$NEW" ]; then
    echo "--- seeding $3 from $2 $(date)" >> "$LOG"
    mkdir -p "$NEW"
    for d in "$SRC"/*/; do
      dn=$(basename "$d"); mkdir -p "$NEW/$dn"
      for f in "$d"*; do [ -e "$f" ] && ln -sf "$(readlink -f "$f")" "$NEW/$dn/$(basename "$f")"; done
    done
    echo "  $3 seeded: $(ls "$NEW/train_img" | wc -l) pairs" >> "$LOG"
  fi
done

stage_one sunny "$CARLA2REAL_DATA/training_pz"       zshr_ || echo "  sunny staging failed" >> "$LOG"
stage_one night "$CARLA2REAL_DATA/training_pz_night" zngt_ || echo "  night staging failed" >> "$LOG"
echo "=== ZOD staging done $(date) ===" >> "$LOG"
