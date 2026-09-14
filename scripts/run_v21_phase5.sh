#!/bin/bash
# v2.1-3: Phase 5 LoRA vs rank-one (matched edit sets, EVAL-only eval)
set -u
PY=/home/shamique/anaconda3/bin/python
CFG=configs/experiment_config.yaml
LOG=logs/v21_phase5.log
echo "==== v2.1 Phase 5 start $(date) ====" | tee -a "$LOG"

$PY src/analysis/phase5_lora_comparison.py --config $CFG \
  --checkpoint-dir outputs/targeted_corrupted \
  --seeds 42 123 456 789 1024 \
  --ranks 1 2 4 8 \
  --layer fc2 \
  --n-examples 100 \
  --epochs 20 \
  --lr 1e-2 \
  --output-dir outputs/phase5 >> "$LOG" 2>&1

echo "==== v2.1 Phase 5 COMPLETE $(date) ====" | tee -a "$LOG"