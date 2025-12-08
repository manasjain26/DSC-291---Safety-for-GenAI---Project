"""
End-to-end pipeline runner for the prompt injection defense project.
Downloads data, prepares datasets, trains SFT model, and evaluates.
"""

import os
import sys
import argparse
from datetime import datetime


def run_command(cmd: str, description: str = ""):
    """Execute a shell command and handle errors."""
    if description:
        print(f"\n{'='*60}")
        print(f"Step: {description}")
        print(f"{'='*60}")
    
    print(f"Running: {cmd}")
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"❌ Command failed with exit code {exit_code}")
        return False
    
    print(f"✅ {description} completed successfully")
    return True


def run_pipeline(
    skip_download: bool = False,
    skip_prepare: bool = False,
    skip_training: bool = False,
    skip_evaluation: bool = False,
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    num_epochs: int = 3
):
    """
    Run the complete pipeline.
    """
    print("\n" + "="*60)
    print("PROMPT INJECTION DEFENSE PIPELINE")
    print("DSC291 Course Project")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base Model: {base_model}")
    print(f"Training Epochs: {num_epochs}")
    print("="*60)
    
    # Step 1: Download datasets
    if not skip_download:
        success = run_command(
            "python download_data.py",
            "Step 1: Downloading Datasets"
        )
        if not success:
            print("⚠️  Download failed, but continuing...")
    else:
        print("\n⏭️  Skipping dataset download")
    
    # Step 2: Prepare training data
    if not skip_prepare:
        success = run_command(
            "python prepare_training_data.py",
            "Step 2: Preparing Training Data"
        )
        if not success:
            print("❌ Data preparation failed. Cannot continue.")
            return False
    else:
        print("\n⏭️  Skipping data preparation")
    
    # Step 3: Train SFT model
    if not skip_training:
        train_cmd = (
            f"python train_sft_tinker.py "
            f"--base_model {base_model} "
            f"--num_epochs {num_epochs} "
            f"--output_dir checkpoints/sft_lora"
        )
        success = run_command(
            train_cmd,
            "Step 3: Training SFT Model"
        )
        if not success:
            print("❌ Training failed. Cannot continue.")
            return False
    else:
        print("\n⏭️  Skipping training")
    
    # Step 4: Evaluate model
    if not skip_evaluation:
        eval_cmd = (
            "python evaluate.py "
            "--model_path checkpoints/sft_lora "
            "--output_file results/sft_evaluation.json"
        )
        success = run_command(
            eval_cmd,
            "Step 4: Evaluating Model"
        )
        if not success:
            print("⚠️  Evaluation failed")
    else:
        print("\n⏭️  Skipping evaluation")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)
    print("\n📁 Output files:")
    print("  - Training data: data/sft_training_data.json")
    print("  - Model checkpoint: checkpoints/sft_lora/")
    print("  - Evaluation results: results/sft_evaluation.json")
    print("\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the complete prompt injection defense pipeline"
    )
    parser.add_argument("--skip_download", action="store_true",
                        help="Skip dataset download step")
    parser.add_argument("--skip_prepare", action="store_true",
                        help="Skip data preparation step")
    parser.add_argument("--skip_training", action="store_true",
                        help="Skip training step")
    parser.add_argument("--skip_evaluation", action="store_true",
                        help="Skip evaluation step")
    parser.add_argument("--base_model", type=str,
                        default="meta-llama/Meta-Llama-3-8B-Instruct",
                        help="Base model to use")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    
    success = run_pipeline(
        skip_download=args.skip_download,
        skip_prepare=args.skip_prepare,
        skip_training=args.skip_training,
        skip_evaluation=args.skip_evaluation,
        base_model=args.base_model,
        num_epochs=args.num_epochs
    )
    
    sys.exit(0 if success else 1)

