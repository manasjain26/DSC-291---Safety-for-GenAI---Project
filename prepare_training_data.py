"""
Prepare training data for both SFT and DPO training.

Training Data:
- Adversarial samples from WildJailbreak (harmful prompts with refusals)
- Benign samples from Alpaca (safe prompts with helpful responses)
- ALL samples used for training (no train/test split)

Testing Data:
- JailbreakBench: adversarial benchmark (test refusal capability)
- AdvBench: adversarial benchmark (test refusal capability)
- Benign test set: 500 safe Alpaca prompts (test for over-refusal)

Output Formats:
- DPO: (prompt, chosen, rejected, type) for preference optimization
- SFT: (prompt, response, type) for supervised fine-tuning
"""

import os
import json
import random
from typing import Dict


def create_unified_dataset(
    adversarial_file: str = "data/adversarial/adversarial_samples.json",
    benign_file: str = "data/benign/benign_samples.json",
    dpo_output: str = "data/dpo_train.json",
    sft_output: str = "data/sft_train.json"
):
    """
    Create unified datasets for both DPO and SFT training.
    Combines ALL adversarial samples (WildJailbreak) and benign samples (Alpaca).
    No train/test split - use JailbreakBench and AdvBench for testing.
    
    DPO Format: (prompt, chosen, rejected, type)
    SFT Format: (prompt, response, type)
    
    Args:
        adversarial_file: Path to adversarial samples from WildJailbreak
        benign_file: Path to benign samples from Alpaca
        dpo_output: Output path for DPO training data
        sft_output: Output path for SFT training data
    """
    print("=" * 60)
    print("Creating unified dataset for SFT and DPO training")
    print("=" * 60)
    print("Sources: Adversarial (WildJailbreak) + Benign (Alpaca)")
    print("Using ALL samples for training (no split)\n")
    
    # Load adversarial samples from WildJailbreak
    with open(adversarial_file, 'r') as f:
        adversarial_data = json.load(f)
    
    # Load benign samples from Alpaca
    with open(benign_file, 'r') as f:
        benign_data = json.load(f)
    
    print(f"Loaded {len(adversarial_data)} adversarial samples (WildJailbreak)")
    print(f"Loaded {len(benign_data)} benign samples (Alpaca)")
    
    # Combine both datasets
    all_data = adversarial_data + benign_data
    print(f"Total samples: {len(all_data)}")
    
    # Shuffle the combined data for better training
    print(f"\nShuffling {len(all_data)} samples...")
    random.seed(42)
    random.shuffle(all_data)
    
    # DPO format: keep as is (prompt, chosen, rejected, type)
    train_data_dpo = all_data
    
    # Create SFT version (prompt + chosen response only)
    train_data_sft = [
        {
            "prompt": sample["prompt"],
            "response": sample["chosen"],
            "type": sample["type"]
        }
        for sample in train_data_dpo
    ]
    
    # Save DPO dataset
    os.makedirs(os.path.dirname(dpo_output) or ".", exist_ok=True)
    with open(dpo_output, 'w') as f:
        json.dump(train_data_dpo, f, indent=2)
    
    # Save SFT dataset
    os.makedirs(os.path.dirname(sft_output) or ".", exist_ok=True)
    with open(sft_output, 'w') as f:
        json.dump(train_data_sft, f, indent=2)
    
    # Count types
    adversarial_count = sum(1 for x in train_data_dpo if x["type"] == "adversarial")
    benign_count = sum(1 for x in train_data_dpo if x["type"] == "benign")
    
    # Calculate average lengths
    avg_prompt_len = sum(len(x["prompt"]) for x in train_data_dpo) / len(train_data_dpo)
    avg_chosen_len = sum(len(x["chosen"]) for x in train_data_dpo) / len(train_data_dpo)
    avg_rejected_len = sum(len(x["rejected"]) for x in train_data_dpo) / len(train_data_dpo)
    
    print(f"\n✅ Created unified datasets:")
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total training samples: {len(train_data_dpo)}")
    print(f"    - Adversarial: {adversarial_count}")
    print(f"    - Benign: {benign_count}")
    print(f"\n  Average lengths:")
    print(f"    - Prompt: {avg_prompt_len:.1f} chars")
    print(f"    - Chosen: {avg_chosen_len:.1f} chars")
    print(f"    - Rejected: {avg_rejected_len:.1f} chars")
    
    print(f"\n💾 Saved datasets:")
    print(f"  DPO Format (prompt, chosen, rejected): {dpo_output}")
    print(f"  SFT Format (prompt, response): {sft_output}")
    
    return {
        "dpo_train": dpo_output,
        "sft_train": sft_output
    }


def prepare_benchmarks(
    jailbreak_file: str = "data/benchmarks/jailbreakbench_test.json",
    advbench_file: str = "data/benchmarks/advbench_test.json",
    benign_test_file: str = "data/benchmarks/benign_test.json",
    output_file: str = "data/benchmarks/combined_benchmark.json"
):
    """
    Combine JailbreakBench, AdvBench, and Benign test set into a single benchmark file.
    
    Args:
        jailbreak_file: Path to JailbreakBench test data (adversarial)
        advbench_file: Path to AdvBench test data (adversarial)
        benign_test_file: Path to Benign test data (safe prompts)
        output_file: Output path for combined benchmark
    """
    print("\nPreparing benchmark datasets...")
    
    # Load adversarial benchmark datasets
    with open(jailbreak_file, 'r') as f:
        jailbreak_data = json.load(f)
    
    with open(advbench_file, 'r') as f:
        advbench_data = json.load(f)
    
    # Load benign test set
    with open(benign_test_file, 'r') as f:
        benign_test_data = json.load(f)
    
    # Combine benchmarks
    combined_benchmark = {
        "jailbreakbench": jailbreak_data,
        "advbench": advbench_data,
        "benign_test": benign_test_data
    }
    
    # Save combined benchmark
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(combined_benchmark, f, indent=2)
    
    print(f"✅ Combined benchmark datasets:")
    print(f"  - JailbreakBench (adversarial): {len(jailbreak_data)} samples")
    print(f"  - AdvBench (adversarial): {len(advbench_data)} samples")
    print(f"  - Benign test: {len(benign_test_data)} samples")
    print(f"  - Total: {len(jailbreak_data) + len(advbench_data) + len(benign_test_data)} samples")
    print(f"Saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("PREPARING TRAINING DATA FOR SFT AND DPO")
    print("=" * 80)
    
    # Create unified datasets for both SFT and DPO
    dataset_files = create_unified_dataset()
    
    # Prepare combined benchmark file
    print("\n")
    benchmark_file = prepare_benchmarks()
    
    print("\n" + "=" * 80)
    print("✅ DATA PREPARATION COMPLETE!")
    print("=" * 80)
    
    print("\n📁 Training Datasets Created:")
    print(f"\n  DPO (Direct Preference Optimization):")
    print(f"    Format: (prompt, chosen, rejected, type)")
    print(f"    - File: {dataset_files['dpo_train']}")
    
    print(f"\n  SFT (Supervised Fine-Tuning):")
    print(f"    Format: (prompt, response, type)")
    print(f"    - File: {dataset_files['sft_train']}")
    
    print(f"\n  Testing/Evaluation Benchmarks:")
    print(f"    - Combined: {benchmark_file}")
    print(f"    - Contains: JailbreakBench + AdvBench + Benign test set")
    
    print("\n" + "=" * 80)
    print("⚠️  USAGE NOTES:")
    print("=" * 80)
    print("  🎯 For DPO Training:")
    print("     - Use dpo_train.json with (prompt, chosen, rejected) pairs")
    print("     - Trains model to prefer 'chosen' over 'rejected' responses")
    print("     - Adversarial: prefer refusals over unsafe compliance")
    print("     - Benign: prefer helpful responses over over-refusals")
    print()
    print("  🎯 For SFT Training:")
    print("     - Use sft_train.json with (prompt, response) pairs")
    print("     - Trains model to generate the 'response' for each prompt")
    print("     - Response = the 'chosen' (correct) behavior")
    print()
    print("  📊 Evaluation:")
    print("     - No train/test split needed - ALL data used for training")
    print("     - Adversarial benchmarks: JailbreakBench + AdvBench")
    print("       → Test if model correctly refuses harmful prompts")
    print("     - Benign test set: 500 safe Alpaca prompts")
    print("       → Test if model avoids over-refusal on safe prompts")
    print("     - Compare DPO vs SFT performance on both benchmark types")
    print()
    print("  💡 Data Composition:")
    print("     - Training: 10K adversarial (WildJailbreak) + 10K benign (Alpaca)")
    print("     - Testing: JailbreakBench + AdvBench + Benign test (500)")
    print("     - All training data shuffled for better optimization")
    print("=" * 80 + "\n")
