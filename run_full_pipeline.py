#!/usr/bin/env python3
"""
Master runner script for the full empathy probe extraction pipeline.

Usage:
    python run_full_pipeline.py [--skip-generation] [--skip-steering]

Options:
    --skip-generation    Skip dataset generation (use existing data)
    --skip-steering      Skip steering experiments (faster)
    --help              Show this help message
"""

import sys
import argparse
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent


def check_dependencies():
    """Check that all required dependencies are installed."""
    logger.info("Checking dependencies...")

    required_packages = [
        'torch',
        'transformers',
        'numpy',
        'sklearn',
        'anthropic',
        'openai'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error(f"Missing required packages: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False

    logger.info("✓ All dependencies installed")
    return True


def step_1_generate_dataset():
    """Step 1: Generate contrastive pairs dataset."""
    logger.info("\n" + "="*80)
    logger.info("STEP 1: GENERATING CONTRASTIVE PAIRS DATASET")
    logger.info("="*80)

    from src.generate_dataset import generate_dataset

    try:
        train_pairs, test_pairs = generate_dataset()
        logger.info(f"✓ Generated {len(train_pairs)} train + {len(test_pairs)} test pairs")
        return True
    except Exception as e:
        logger.error(f"✗ Dataset generation failed: {e}")
        return False


def step_2_extract_probes():
    """Step 2: Extract empathy probes."""
    logger.info("\n" + "="*80)
    logger.info("STEP 2: EXTRACTING EMPATHY PROBES")
    logger.info("="*80)

    from src.probe_extraction import main as extract_main

    try:
        extract_main()
        logger.info("✓ Probe extraction complete")
        return True
    except Exception as e:
        logger.error(f"✗ Probe extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_3_predict_eia_scores():
    """Step 3: Predict EIA behavioral scores."""
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PREDICTING EIA BEHAVIORAL SCORES")
    logger.info("="*80)

    from src.eia_evaluator import main as eia_main

    try:
        eia_main()
        logger.info("✓ EIA score prediction complete")
        return True
    except Exception as e:
        logger.error(f"✗ EIA prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def step_4_steering_experiments(skip: bool = False):
    """Step 4: Run steering experiments."""
    if skip:
        logger.info("\n" + "="*80)
        logger.info("STEP 4: STEERING EXPERIMENTS (SKIPPED)")
        logger.info("="*80)
        return True

    logger.info("\n" + "="*80)
    logger.info("STEP 4: RUNNING STEERING EXPERIMENTS")
    logger.info("="*80)

    from src.steering import main as steering_main

    try:
        steering_main()
        logger.info("✓ Steering experiments complete")
        return True
    except Exception as e:
        logger.error(f"✗ Steering experiments failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_summary():
    """Print final summary of results."""
    import json

    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE - FINAL SUMMARY")
    logger.info("="*80)

    results_dir = PROJECT_ROOT / "results"

    # Validation results
    validation_path = results_dir / "validation_auroc.json"
    if validation_path.exists():
        with open(validation_path) as f:
            val_results = json.load(f)

        logger.info("\n📊 VALIDATION RESULTS:")
        logger.info(f"  Best Layer: {val_results['best_layer']}")
        logger.info(f"  Best AUROC: {val_results['best_auroc']:.4f}")
        logger.info(f"  Target >0.75: {'✓ ACHIEVED' if val_results['best_auroc'] >= 0.75 else '✗ Not reached'}")

    # EIA correlation results
    eia_path = results_dir / "eia_correlation.json"
    if eia_path.exists():
        with open(eia_path) as f:
            eia_results = json.load(f)

        logger.info("\n🎯 EIA SCORE PREDICTION:")
        logger.info(f"  Pearson correlation: {eia_results['pearson_correlation']:.4f}")
        logger.info(f"  Binary accuracy: {eia_results['binary_accuracy']:.4f}")
        logger.info(f"  Target r>0.4: {'✓ ACHIEVED' if abs(eia_results['pearson_correlation']) >= 0.4 else '✗ Not reached'}")

    # Steering results
    steering_path = results_dir / "steering_examples.json"
    if steering_path.exists():
        with open(steering_path) as f:
            steering_results = json.load(f)

        logger.info("\n🎛️ STEERING EXPERIMENTS:")
        logger.info(f"  Scenarios tested: {len(steering_results['experiments'])}")
        logger.info(f"  Alpha values: {steering_results['alphas_tested']}")
        logger.info("  See results/steering_examples.json for detailed comparisons")

    logger.info("\n" + "="*80)
    logger.info("All results saved to: results/")
    logger.info("  - validation_auroc.json")
    logger.info("  - eia_correlation.json")
    logger.info("  - steering_examples.json")
    logger.info("  - probes/empathy_direction_layer_*.npy")
    logger.info("="*80)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run full empathy probe extraction pipeline"
    )
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip dataset generation (use existing data)'
    )
    parser.add_argument(
        '--skip-steering',
        action='store_true',
        help='Skip steering experiments'
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        sys.exit(1)

    # Run pipeline
    logger.info("Starting empathy probe extraction pipeline...")
    logger.info(f"Project root: {PROJECT_ROOT}")

    success = True

    # Step 1: Generate dataset
    if not args.skip_generation:
        if not step_1_generate_dataset():
            logger.error("Pipeline failed at dataset generation")
            sys.exit(1)
    else:
        logger.info("Skipping dataset generation (using existing data)")

    # Step 2: Extract probes
    if not step_2_extract_probes():
        logger.error("Pipeline failed at probe extraction")
        sys.exit(1)

    # Step 3: Predict EIA scores
    if not step_3_predict_eia_scores():
        logger.error("Pipeline failed at EIA prediction")
        sys.exit(1)

    # Step 4: Steering experiments
    if not step_4_steering_experiments(skip=args.skip_steering):
        logger.warning("Steering experiments failed (non-critical)")

    # Print summary
    print_summary()

    logger.info("\n✓ Pipeline completed successfully!")


if __name__ == "__main__":
    main()
