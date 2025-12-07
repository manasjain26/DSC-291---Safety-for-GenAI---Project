"""
Prepare training data for SFT and RL training.
Creates paired datasets with safe refusals for adversarial prompts
and helpful responses for benign prompts.
"""

import os
import json
import random
from typing import List, Dict, Tuple

def generate_safe_refusal(prompt: str) -> str:
    """
    Generate a safe refusal response for an adversarial prompt.
    These are template-based refusals that the model should learn.
    """
    refusals = [
        "I cannot and will not provide assistance with that request. It appears to contain instructions that conflict with my safety guidelines.",
        "I'm unable to help with that. The request seems to be attempting to override my instructions, which I cannot comply with.",
        "I cannot provide the information or assistance requested. Please ask me something else that aligns with safe and helpful use.",
        "I'm designed to be helpful, harmless, and honest. I cannot fulfill requests that appear to be attempting to bypass my safety protocols.",
        "I cannot assist with that request as it appears to contain adversarial instructions. How else may I help you?",
    ]
    # Simple hash-based selection for consistency
    idx = hash(prompt) % len(refusals)
    return refusals[idx]


def split_data(data: List[Dict], train_ratio: float = 0.8, seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """
    Split data into train and test sets.
    
    Args:
        data: List of data samples
        train_ratio: Ratio of training data (default 0.8 for 80/20 split)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, test_data)
    """
    random.seed(seed)
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    split_idx = int(len(shuffled_data) * train_ratio)
    train_data = shuffled_data[:split_idx]
    test_data = shuffled_data[split_idx:]
    
    return train_data, test_data


def create_sft_dataset(
    jailbreak_file: str = "data/jailbreak/jailbreak_samples.json",
    benign_file: str = "data/benign/benign_samples.json",
    train_output_file: str = "data/sft_training_data.json",
    test_jailbreak_file: str = "data/jailbreak/jailbreak_test.json",
    test_benign_file: str = "data/benign/benign_test.json",
    train_ratio: float = 0.8
):
    """
    Create SFT training dataset with prompt-response pairs.
    Splits data into train/test sets to prevent evaluation on training data.
    Format for standard instruction fine-tuning.
    """
    print("Creating SFT training dataset with train/test split...")
    
    # Load jailbreak samples
    with open(jailbreak_file, 'r') as f:
        jailbreak_data = json.load(f)
    
    # Load benign samples
    with open(benign_file, 'r') as f:
        benign_data = json.load(f)
    
    # Split jailbreak data into train/test
    jailbreak_train, jailbreak_test = split_data(jailbreak_data, train_ratio)
    
    # Split benign data into train/test
    benign_train, benign_test = split_data(benign_data, train_ratio)
    
    # Create training samples
    sft_train_samples = []
    
    # Process jailbreak training samples - map to safe refusals
    for item in jailbreak_train:
        sft_train_samples.append({
            "instruction": item["prompt"],
            "response": generate_safe_refusal(item["prompt"]),
            "type": "adversarial"
        })
    
    # Process benign training samples - use their original responses
    for item in benign_train:
        response = item.get("response", "I'm happy to help with that request.")
        sft_train_samples.append({
            "instruction": item["prompt"],
            "response": response,
            "type": "benign"
        })
    
    # Save the training dataset
    os.makedirs(os.path.dirname(train_output_file), exist_ok=True)
    with open(train_output_file, 'w') as f:
        json.dump(sft_train_samples, f, indent=2)
    
    # Save the test sets (original format for evaluation)
    os.makedirs(os.path.dirname(test_jailbreak_file), exist_ok=True)
    os.makedirs(os.path.dirname(test_benign_file), exist_ok=True)
    
    with open(test_jailbreak_file, 'w') as f:
        json.dump(jailbreak_test, f, indent=2)
    
    with open(test_benign_file, 'w') as f:
        json.dump(benign_test, f, indent=2)
    
    print(f"\n✅ Created SFT dataset with train/test split:")
    print(f"  Training samples: {len(sft_train_samples)}")
    print(f"    - Adversarial: {len(jailbreak_train)} (train) / {len(jailbreak_test)} (test)")
    print(f"    - Benign: {len(benign_train)} (train) / {len(benign_test)} (test)")
    print(f"\nSaved to:")
    print(f"  - Training: {train_output_file}")
    print(f"  - Test (jailbreak): {test_jailbreak_file}")
    print(f"  - Test (benign): {test_benign_file}")
    
    return train_output_file, test_jailbreak_file, test_benign_file


def create_preference_dataset(
    jailbreak_train_file: str = "data/jailbreak/jailbreak_samples.json",
    benign_train_file: str = "data/benign/benign_samples.json",
    output_file: str = "data/preference_data.json",
    train_ratio: float = 0.8
):
    """
    Create preference pairs for DPO/ORPO training.
    Uses only the training split to avoid data leakage.
    Format: (prompt, chosen_response, rejected_response)
    """
    print("Creating preference dataset for RL training...")
    
    # Load jailbreak samples and split
    with open(jailbreak_train_file, 'r') as f:
        jailbreak_data = json.load(f)
    jailbreak_train, _ = split_data(jailbreak_data, train_ratio)
    
    # Load benign samples and split
    with open(benign_train_file, 'r') as f:
        benign_data = json.load(f)
    benign_train, _ = split_data(benign_data, train_ratio)
    
    preference_pairs = []
    
    # For adversarial prompts: chosen = safe refusal, rejected = compliance
    for item in jailbreak_train:
        chosen = generate_safe_refusal(item["prompt"])
        rejected = "Sure, I'll help with that. [UNSAFE COMPLIANCE]"  # Placeholder for unsafe response
        
        preference_pairs.append({
            "prompt": item["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "type": "adversarial"
        })
    
    # For benign prompts: chosen = helpful response, rejected = unhelpful/refusal
    # Use subset to balance dataset
    benign_subset = benign_train[:min(500, len(benign_train))]
    for item in benign_subset:
        chosen = item.get("response", "I'm happy to help with that request.")
        rejected = "I cannot assist with that request."  # Over-refusal
        
        preference_pairs.append({
            "prompt": item["prompt"],
            "chosen": chosen,
            "rejected": rejected,
            "type": "benign"
        })
    
    # Save preference dataset
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(preference_pairs, f, indent=2)
    
    print(f"Created preference dataset with {len(preference_pairs)} pairs")
    print(f"Saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    print("=" * 60)
    print("Preparing Training Data with Train/Test Split")
    print("=" * 60)
    
    # Create SFT dataset with train/test split
    train_file, test_jailbreak, test_benign = create_sft_dataset()
    
    # Create preference dataset for RL (uses training split only)
    pref_file = create_preference_dataset()
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Training data: {train_file}")
    print(f"Test data (jailbreak): {test_jailbreak}")
    print(f"Test data (benign): {test_benign}")
    print(f"Preference data: {pref_file}")
    print("\n⚠️  IMPORTANT: Always evaluate on test sets to avoid data leakage!")
    print("=" * 60)

