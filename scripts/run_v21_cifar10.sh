#!/bin/bash
# v2.1-1: CIFAR-10 replication under v2 protocol (5 config seeds)
set -u
PY=/home/shamique/anaconda3/bin/python
CFG=configs/experiment_config.yaml
SEEDS="42 123 456 789 1024"
LOG=logs/v21_cifar10.log
echo "==== v2.1 CIFAR-10 start $(date) ====" | tee -a "$LOG"

for s in $SEEDS; do
  [ -f "outputs/cifar10/clean/seed_$s/final_model.pt" ] || \
    $PY src/training/train_cifar10_clean.py --config $CFG --seed $s --output-dir outputs/cifar10/clean >> "$LOG" 2>&1
  echo "clean seed $s done" >> "$LOG"
done
for s in $SEEDS; do
  [ -f "outputs/cifar10/corrupted/noise_0.2/seed_$s/final_model.pt" ] || \
    $PY src/training/train_cifar10_corrupted.py --config $CFG --seed $s --noise-rate 0.2 --output-dir outputs/cifar10/corrupted >> "$LOG" 2>&1
  echo "corrupted seed $s done" >> "$LOG"
done

# Analyses (v2)
$PY src/analysis/analyze_cifar10.py --config $CFG \
  --clean-dir outputs/cifar10/clean --corrupted-dir outputs/cifar10/corrupted \
  --seeds $SEEDS --output-dir outputs/cifar10/analysis >> "$LOG" 2>&1
$PY src/analysis/rome_cifar10.py --config $CFG \
  --clean-dir outputs/cifar10/clean --corrupted-dir outputs/cifar10/corrupted \
  --seeds $SEEDS --output-dir outputs/cifar10/analysis/rome >> "$LOG" 2>&1
$PY src/analysis/cifar_replication.py \
  --clean-dir outputs/cifar10/clean --corrupted-dir outputs/cifar10/corrupted \
  --seeds $SEEDS --output-dir outputs/cifar10/replication >> "$LOG" 2>&1

echo "==== v2.1 CIFAR-10 COMPLETE $(date) ====" | tee -a "$LOG"
