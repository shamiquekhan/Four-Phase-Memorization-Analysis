#!/usr/bin/env python3
"""
Single script to reproduce the full memorization analysis pipeline.
"""
import argparse
import subprocess
import sys
from pathlib import Path
import time


def run(cmd, desc, cwd=None):
    print(f"\n{'='*60}")
    print(f"Step: {desc}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, shell=True, cwd=cwd or '.')
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"ERROR: {desc} failed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"Completed in {elapsed:.1f}s")
    return result


def main():
    parser = argparse.ArgumentParser(description='Reproduce full memorization analysis pipeline')
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml')
    parser.add_argument('--seeds', type=int, default=10, help='Number of seeds')
    parser.add_argument('--skip-training', action='store_true', help='Skip training (use existing checkpoints)')
    parser.add_argument('--skip-scaling', action='store_true', help='Skip scaling experiments')
    parser.add_argument('--skip-analysis', action='store_true', help='Skip analysis')
    parser.add_argument('--skip-figures', action='store_true', help='Skip figure generation')
    parser.add_argument('--skip-cifar10', action='store_true', help='Skip CIFAR-10 validation experiments')
    parser.add_argument('--phase5', action='store_true', help='Run Phase 5: LoRA vs ROME subspace comparison')
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    
    root = Path(__file__).parent
    seeds = list(range(args.seeds))
    
    # Step 1: Train on clean MNIST
    if not args.skip_training:
        for seed in seeds:
            run(
                f"python src/training/train_clean.py --config {args.config} --seed {seed} --output-dir {args.output_dir}/clean",
                f"Training clean model (seed {seed})",
                cwd=root
            )
        
        # Step 2: Train on corrupted MNIST
        for seed in seeds:
            run(
                f"python src/training/train_corrupted.py --config {args.config} --seed {seed} --noise-rate 0.2 --output-dir {args.output_dir}/corrupted",
                f"Training corrupted model (seed {seed})",
                cwd=root
            )
        
        # Step 3: Scaling experiment
        if not args.skip_scaling:
            run(
                f"python src/scaling/train_scaling.py --config {args.config} --hidden-dims 32 64 128 256 512 1024 --seeds {' '.join(map(str, seeds))} --epochs 20 --output-dir {args.output_dir}/scaling",
                "Running scaling experiments",
                cwd=root
            )
        
        # Step 3b: CIFAR-10 validation experiments
        if not args.skip_cifar10:
            cifar_seeds = seeds[:5]
            for seed in cifar_seeds:
                run(
                    f"python src/training/train_cifar10_clean.py --config {args.config} --seed {seed} --output-dir {args.output_dir}/cifar10/clean",
                    f"Training CIFAR-10 clean model (seed {seed})",
                    cwd=root
                )
            for seed in cifar_seeds:
                run(
                    f"python src/training/train_cifar10_corrupted.py --config {args.config} --seed {seed} --noise-rate 0.2 --output-dir {args.output_dir}/cifar10/corrupted",
                    f"Training CIFAR-10 corrupted model (seed {seed})",
                    cwd=root
                )
            for hdim in [64, 128, 256, 512]:
                for seed in cifar_seeds[:3]:
                    run(
                        f"python src/training/train_cifar10_scaling.py --config {args.config} --seed {seed} --hidden-dim {hdim} --output-dir {args.output_dir}/cifar10/scaling",
                        f"Training CIFAR-10 scaled MLP h={hdim} (seed {seed})",
                        cwd=root
                    )
    
    # Step 4: Phase 1 - Basic analysis
    if not args.skip_analysis:
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/phase1_basic.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds))} --output-dir {args.output_dir}/analysis/phase1_{tag}",
                f"Phase 1 analysis ({tag})",
                cwd=root
            )
        
        # Step 5: Phase 2 - Representation analysis
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/phase2_representation.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds))} --output-dir {args.output_dir}/analysis/phase2_{tag}",
                f"Phase 2 analysis ({tag})",
                cwd=root
            )
        
        # Step 6: Phase 3 - Influence functions
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/phase3_influence.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds[:5]))} --output-dir {args.output_dir}/analysis/phase3_{tag}",
                f"Phase 3 analysis ({tag})",
                cwd=root
            )
        
        # Step 7: Phase 4 - ROME analysis
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/phase4_rome.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds))} --output-dir {args.output_dir}/analysis/phase4_{tag}",
                f"Phase 4 analysis ({tag})",
                cwd=root
            )
        
        # Step 8: Multi-class ROME validation
        for tag, checkpoint_dir in [('targeted_corrupted', f'{args.output_dir}/targeted_corrupted')]:
            run(
                f"python src/analysis/multiclass_rome.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds[:5]))} --output-dir {args.output_dir}/analysis/multiclass_rome_{tag}",
                f"Multi-class ROME validation ({tag})",
                cwd=root
            )
        
        # Step 9: Rank ablation
        for tag, checkpoint_dir in [('clean', f'{args.output_dir}/clean'), ('corrupted', f'{args.output_dir}/corrupted/noise_0.2')]:
            run(
                f"python src/analysis/rank_ablation.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds[:5]))} --output-dir {args.output_dir}/analysis/rank_ablation_{tag}",
                f"Rank ablation study ({tag})",
                cwd=root
            )
        
        # Step 10: Scaling analysis
        if not args.skip_scaling:
            run(
                f"python src/scaling/analyze_scaling.py --results-dir {args.output_dir}/scaling --output-dir {args.output_dir}/analysis/scaling",
                "Scaling analysis",
                cwd=root
            )
        
        # Step 10b: CIFAR-10 validation analysis
        if not args.skip_cifar10:
            cifar_seeds = ' '.join(map(str, seeds[:5]))
            run(
                f"python src/analysis/analyze_cifar10.py --clean-dir {args.output_dir}/cifar10/clean --corrupted-dir {args.output_dir}/cifar10/corrupted --seeds {cifar_seeds} --output-dir {args.output_dir}/cifar10/analysis",
                "CIFAR-10 Phase 1+2 analysis",
                cwd=root
            )
            run(
                f"python src/analysis/rome_cifar10.py --clean-dir {args.output_dir}/cifar10/clean --corrupted-dir {args.output_dir}/cifar10/corrupted --seeds {cifar_seeds} --output-dir {args.output_dir}/cifar10/analysis/rome",
                "CIFAR-10 ROME validation",
                cwd=root
            )
            run(
                f"python src/analysis/analyze_cifar10_scaling.py --checkpoint-dir {args.output_dir}/cifar10/scaling --hidden-dims 64 128 256 512 --seeds {' '.join(map(str, seeds[:3]))} --output-dir {args.output_dir}/cifar10/analysis/scaling",
                "CIFAR-10 scaling analysis",
                cwd=root
            )
    
    # Step 11: Generate figures
    if not args.skip_figures:
        run(
            f"python src/analysis/visualizations.py --results-dir {args.output_dir} --output-dir {args.output_dir}/figures --config {args.config}",
            "Generating figures",
            cwd=root
        )
    
    # Step 12: Run verification scripts
    run("python scripts/verify_consistency.py", "Verifying cross-document consistency", cwd=root)
    run("python scripts/verify_statistics.py", "Verifying statistical claims", cwd=root)
    run("python scripts/update_readme.py", "Updating README with latest results", cwd=root)

    # Step 13: Run unit tests
    run(
        "python -m pytest tests/ -v --tb=short",
        "Running unit tests",
        cwd=root
    )

    # Step 14: Multi-layer ROME
    for tag, checkpoint_dir in [('targeted_corrupted', f'{args.output_dir}/targeted_corrupted')]:
        run(
            f"python src/analysis/multilayer_rome.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds[:5]))} --output-dir {args.output_dir}/analysis/multilayer_rome",
            "Multi-layer ROME (sequential + joint)",
            cwd=root
        )

    # Step 14b: Random baseline for ROME
    for tag, checkpoint_dir in [('targeted_corrupted', f'{args.output_dir}/targeted_corrupted')]:
        run(
            f"python src/analysis/multiclass_rome.py --config {args.config} --checkpoint-dir {checkpoint_dir} --seeds {' '.join(map(str, seeds[:5]))} --output-dir {args.output_dir}/analysis/multiclass_rome_{tag}",
            "Multi-class ROME with random baseline",
            cwd=root
        )

    # Step 15: CIFAR-10 replication (CKA, ROME, rank ablation)
    if not args.skip_cifar10:
        run(
            f"python src/analysis/cifar_replication.py --clean-dir {args.output_dir}/cifar10/clean --corrupted-dir {args.output_dir}/cifar10/corrupted --seeds {' '.join(map(str, seeds[:5]))} --output-dir {args.output_dir}/cifar10/replication",
            "CIFAR-10 replication (CKA + ROME + rank ablation)",
            cwd=root
        )

    # Step 16: Phase 5 - LoRA vs ROME Subspace Comparison
    if args.phase5:
        # Use all 10 seeds for Phase 5 to match paper
        run(
            f"python src/analysis/phase5_lora_comparison.py --config {args.config} --checkpoint-dir {args.output_dir}/targeted_corrupted --seeds {' '.join(map(str, seeds))} --output-dir {args.output_dir}/phase5 --layer fc2 --ranks 1 2 4 8 --n-examples 100 --epochs 20 --lr 1e-2",
            "Phase 5: LoRA vs ROME subspace comparison",
            cwd=root
        )
    
    # Step 17: Final verification
    run("python scripts/verify_consistency.py", "Final consistency check", cwd=root)
    
    print(f"\n{'='*60}")
    print("Pipeline complete!")
    print(f"{'='*60}")
    print(f"All outputs in: {args.output_dir}")
    print(f"Run `python -m pytest tests/ -v` to verify results.")


if __name__ == '__main__':
    main()