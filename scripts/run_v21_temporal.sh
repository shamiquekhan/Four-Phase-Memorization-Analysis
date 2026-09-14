#!/bin/bash
# v2.1-4: Temporal per-epoch checkpoint dynamics
set -u
PY=/home/shamique/anaconda3/bin/python
CFG=configs/experiment_config.yaml
LOG=logs/v21_temporal.log
echo "==== v2.1 Temporal start $(date) ====" | tee -a "$LOG"

# Train clean (5 seeds)
for s in 42 123 456 789 1024; do
    $PY src/training/train_temporal.py --config $CFG --seed $s --noise-rate 0.0 --output-dir outputs/temporal >> "$LOG" 2>&1
    echo "clean temporal seed $s done" >> "$LOG"
done

# Train corrupted 20% (5 seeds)
for s in 42 123 456 789 1024; do
    $PY src/training/train_temporal.py --config $CFG --seed $s --noise-rate 0.2 --output-dir outputs/temporal >> "$LOG" 2>&1
    echo "corr temporal seed $s done" >> "$LOG"
done

echo "==== v2.1 Temporal COMPLETE $(date) ====" | tee -a "$LOG"