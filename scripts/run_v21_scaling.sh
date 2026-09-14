#!/bin/bash
# v2.1-2: Scaling sweep under v2 protocol (clean + corrupted, 5 config seeds)
# Doubles as v2.1-5 (memorization regime): behavioral audit is saved per run,
# so memorized-fraction-vs-width is read directly from the results.
set -u
PY=/home/shamique/anaconda3/bin/python
CFG=configs/experiment_config.yaml
SEEDS="42 123 456 789 1024"
WIDTHS="16 32 64 128 256 512 1024"
LOG=logs/v21_scaling.log
echo "==== v2.1 scaling start $(date) ====" | tee -a "$LOG"

# Clean sweep (epochs=20 to match primary protocol, not the legacy 100)
for w in $WIDTHS; do
  for s in $SEEDS; do
    [ -f "outputs/scaling/clean/hidden_$w/seed_$s/final_model.pt" ] && continue
    $PY src/scaling/train_scaling.py --config $CFG --hidden-dims $w --seeds $s \
      --epochs 20 --noise-rate 0.0 --output-dir outputs/scaling >> "$LOG" 2>&1 \
      && echo "clean w=$w s=$s done" >> "$LOG" || { echo "FAILED clean w=$w s=$s" >> "$LOG"; exit 1; }
  done
done

# Corrupted sweep (same budget; behavioral audit saved per run)
for w in $WIDTHS; do
  for s in $SEEDS; do
    [ -f "outputs/scaling/noise_0.2/hidden_$w/seed_$s/final_model.pt" ] && continue
    $PY src/scaling/train_scaling.py --config $CFG --hidden-dims $w --seeds $s \
      --epochs 20 --noise-rate 0.2 --output-dir outputs/scaling >> "$LOG" 2>&1 \
      && echo "corr w=$w s=$s done" >> "$LOG" || { echo "FAILED corr w=$w s=$s" >> "$LOG"; exit 1; }
  done
done

echo "==== v2.1 scaling COMPLETE $(date) ====" | tee -a "$LOG"
