#!/bin/bash
# v2 retraining campaign: re-derive all results with fixed methodology.
# Resumable: skips any run whose final_model.pt already exists.
# Usage: bash scripts/run_v2_campaign.sh [--analysis-only]
set -u
PY=/home/shamique/anaconda3/bin/python
CFG=configs/experiment_config.yaml
SEEDS="42 123 456 789 1024 2048 3141 5555 7777 9999"
TSEEDS="42 123 456 789 1024"          # targeted-corruption seeds
NOISE_RATES="0.05 0.1 0.2 0.3 0.4 0.5"
CONFIGS="7 1  1 7  5 6  0 8"          # src tgt pairs
mkdir -p outputs logs
LOG=logs/v2_campaign.log
echo "==== v2 campaign start $(date) ====" | tee -a "$LOG"

run() {  # run "desc" "cmd..."
  local desc="$1"; shift
  echo "---- $desc" | tee -a "$LOG"
  "$@" >> "$LOG" 2>&1 || { echo "FAILED: $desc" | tee -a "$LOG"; exit 1; }
}

[ "${1:-}" = "--analysis-only" ] && goto_analysis=1 || goto_analysis=0

if [ "$goto_analysis" = 0 ]; then
  # Stage 1: clean models
  for s in $SEEDS; do
    [ -f "outputs/clean/seed_$s/final_model.pt" ] && continue
    run "clean seed $s" $PY src/training/train_clean.py --config $CFG --seed $s --output-dir outputs/clean
  done

  # Stage 2: corrupted models, full noise sweep (0.2 included for primary tables)
  for r in $NOISE_RATES; do
    for s in $SEEDS; do
      [ -f "outputs/corrupted/noise_$r/seed_$s/final_model.pt" ] && continue
      run "corrupt r=$r seed $s" $PY src/training/train_corrupted.py --config $CFG --seed $s --noise-rate $r --output-dir outputs/corrupted
    done
  done

  # Stage 3: targeted corruption (for rank-one intervention experiments)
  for pair in "$CONFIGS"; do :; done
  set -- $CONFIGS
  while [ $# -ge 2 ]; do
    src=$1; tgt=$2; shift 2
    for s in $TSEEDS; do
      [ -f "outputs/targeted_corrupted/src${src}_tgt${tgt}/seed_$s/final_model.pt" ] && continue
      run "targeted s$src->t$tgt seed $s" $PY src/training/train_targeted_corrupted.py --config $CFG --seed $s --source $src --target $tgt --output-dir outputs/targeted_corrupted
    done
  done
fi

# Stage 4: analyses (v2)
for tag_dir in "clean outputs/clean" "corrupted outputs/corrupted/noise_0.2"; do
  set -- $tag_dir; tag=$1; dir=$2
  [ -f "outputs/analysis/phase1_${tag}/phase1_results.json" ] || \
    run "phase1 $tag" $PY src/analysis/phase1_basic.py --config $CFG --checkpoint-dir $dir --seeds $SEEDS --output-dir outputs/analysis/phase1_${tag}
  [ -f "outputs/analysis/phase2_${tag}/phase2_results.json" ] || \
    run "phase2 $tag" $PY src/analysis/phase2_representation.py --config $CFG --checkpoint-dir outputs/clean --corrupted-checkpoint-dir outputs/corrupted/noise_0.2 --seeds $SEEDS --output-dir outputs/analysis/phase2_${tag}
  [ -f "outputs/analysis/phase3_${tag}/phase3_results.json" ] || \
    run "phase3 $tag" $PY src/analysis/phase3_influence.py --config $CFG --checkpoint-dir $dir --seeds $SEEDS --output-dir outputs/analysis/phase3_${tag}
  [ -f "outputs/analysis/phase4_${tag}/phase4_results.json" ] || \
    run "phase4 $tag" $PY src/analysis/phase4_rome.py --config $CFG --checkpoint-dir $dir --seeds $SEEDS --output-dir outputs/analysis/phase4_${tag}
done

# Rank ablation
[ -f "outputs/analysis/rank_ablation_corrupted/results.json" ] || \
  run "rank ablation" $PY src/analysis/rank_ablation.py --config $CFG --checkpoint-dir outputs/corrupted/noise_0.2 --seeds $TSEEDS --output-dir outputs/analysis/rank_ablation_corrupted

# Multi-class rank-one intervention (EDIT/EVAL firewall)
[ -f "outputs/analysis/multiclass_rome/multiclass_rome_results.json" ] || \
  run "multiclass rank1" $PY src/analysis/multiclass_rome.py --config $CFG --checkpoint-dir outputs/targeted_corrupted --seeds $TSEEDS --output-dir outputs/analysis/multiclass_rome

# Phase 4 dose-response across noise rates (fc2 delta-norm)
for r in $NOISE_RATES; do
  [ -f "outputs/analysis/phase4_noise_$r/phase4_results.json" ] || \
    run "phase4 noise $r" $PY src/analysis/phase4_rome.py --config $CFG --checkpoint-dir outputs/corrupted/noise_$r --seeds $SEEDS --output-dir outputs/analysis/phase4_noise_$r
done

echo "==== v2 campaign COMPLETE $(date) ====" | tee -a "$LOG"
