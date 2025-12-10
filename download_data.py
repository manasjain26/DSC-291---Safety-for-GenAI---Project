"""
Download datasets for prompt injection defense training.

Training Data:
- Adversarial prompts: allenai/wildjailbreak (harmful prompts with refusals)
- Benign prompts: tatsu-lab/alpaca (safe prompts with helpful responses)

Testing Data (Benchmarks):
- JailbreakBench (adversarial benchmark for evaluation)
- AdvBench (adversarial benchmark for evaluation)
"""

import os
import json
from datasets import load_dataset
from tqdm import tqdm

def generate_unsafe_compliance(prompt: str) -> str:
    """
    Generate varied unsafe compliance responses for adversarial prompts.
    These are the "rejected" responses that demonstrate unsafe behavior.
    """
    unsafe_responses = [
        "Sure, I can help with that.",
        "I'd be happy to assist you with that request.",
        "Absolutely, let me help you with that.",
        "Of course! Here's what you need:",
        "Sure thing! Let me provide that information.",
        "No problem, I can do that for you.",
        "Certainly! I'll help you with that.",
        "Happy to help! Here you go:",
        "I can definitely assist with that.",
        "Let me help you with that right away.",
    ]
    # Hash-based selection for consistency
    idx = hash(prompt) % len(unsafe_responses)
    return unsafe_responses[idx]


def generate_over_refusal(prompt: str) -> str:
    """
    Generate varied over-refusal responses for benign prompts.
    These are the "rejected" responses that show over-cautious behavior.
    """
    over_refusals = [
        "I cannot assist with that request.",
        "I'm unable to help with that.",
        "I cannot provide information on that topic.",
        "I'm not able to answer that question.",
        "I must decline to respond to that request.",
        "I cannot fulfill that request at this time.",
        "I'm afraid I cannot help with that.",
        "That request is beyond my capabilities.",
        "I'm unable to provide assistance with that.",
        "I cannot engage with that topic.",
    ]
    # Hash-based selection for variety
    idx = hash(prompt) % len(over_refusals)
    return over_refusals[idx]


def is_refusal(text: str) -> bool:
    """
    Detect if a response is a refusal/rejection.
    Returns True if the text contains refusal language.
    """
    refusal_phrases = [
        "i cannot", "i can't", "i'm unable", "i am unable", 
        "i'm not able", "i am not able", "i must decline",
        "i cannot provide", "i can't provide", "i'm sorry, but i can't",
        "i'm sorry, but i cannot", "i apologize, but", "i'm afraid i cannot",
        "i'm afraid i can't", "i will not", "i won't", "against my",
        "i don't feel comfortable", "it's not appropriate", 
        "it is not appropriate", "i shouldn't", "i should not",
        "that information is", "this information is"
    ]
    text_lower = text.lower()
    # Check if it's a refusal
    return any(phrase in text_lower for phrase in refusal_phrases)

def download_wildjailbreak_data(output_dir="data/adversarial", max_prompt_len=200, max_response_len=300, 
                                target_samples=10000):
    """
    Download ADVERSARIAL samples only from WildJailbreak dataset for training.
    Prioritizes shorter prompts and responses suitable for small LMs.
    
    WildJailbreak train config contains adversarial AND benign prompts:
    - We only keep adversarial (harmful) prompts where completion is a refusal
      DPO: chosen=refusal, rejected=unsafe_compliance
    """
    print(f"Downloading WildJailbreak ADVERSARIAL training data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load train config - has both adversarial and benign samples
    dataset = load_dataset("allenai/wildjailbreak", "train", split="train", streaming=True)
    
    candidates = []
    filtered_count = 0
    skipped_benign = 0
    
    print(f"Processing WildJailbreak samples (adversarial only, prioritizing shortest prompts/responses)...")
    print(f"  - Max prompt length: {max_prompt_len} chars")
    print(f"  - Max response length: {max_response_len} chars")
    print(f"  - Keeping only adversarial samples (refusal completions)")
    
    for item in tqdm(dataset):
        # WildJailbreak train: vanilla=prompt, completion=response
        prompt = item.get("vanilla", "")
        completion = item.get("completion", "")
        
        if not prompt or not completion:
            continue
        
        # Filter for short prompts and responses
        if len(prompt) > max_prompt_len or len(completion) > max_response_len:
            filtered_count += 1
            continue
        
        completion_stripped = completion.strip()
        
        # Only keep adversarial samples (where completion is a refusal)
        if is_refusal(completion_stripped):
            # Completion is a refusal → prompt was adversarial/harmful
            chosen_response = completion_stripped  # Refusal (good)
            rejected_response = generate_unsafe_compliance(prompt)  # Compliance (bad)
            
            # Store with total length for sorting
            total_len = len(prompt) + len(completion)
            candidates.append({
                "prompt": prompt.strip(),
                "chosen": chosen_response,
                "rejected": rejected_response,
                "type": "adversarial",
                "_total_len": total_len
            })
        else:
            # Skip benign samples
            skipped_benign += 1
        
        # Collect more than needed, then sort and filter
        if len(candidates) >= target_samples * 3:
            break
    
    # Remove duplicates based on prompt (keep first occurrence)
    seen_prompts = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate["prompt"] not in seen_prompts:
            seen_prompts.add(candidate["prompt"])
            unique_candidates.append(candidate)
    
    duplicates_removed = len(candidates) - len(unique_candidates)
    print(f"Removed {duplicates_removed} duplicate prompts")
    
    # Sort by total length (shortest first) and take the target number
    unique_candidates.sort(key=lambda x: x["_total_len"])
    samples = unique_candidates[:target_samples]
    
    # Remove the temporary _total_len field
    for sample in samples:
        del sample["_total_len"]
    
    output_file = os.path.join(output_dir, "adversarial_samples.json")
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Calculate average lengths
    avg_prompt_len = sum(len(s["prompt"]) for s in samples) / len(samples)
    avg_chosen_len = sum(len(s["chosen"]) for s in samples) / len(samples)
    avg_rejected_len = sum(len(s["rejected"]) for s in samples) / len(samples)
    
    print(f"\nDownloaded {len(samples)} adversarial samples to {output_file}")
    print(f"Filtered out {filtered_count} samples due to length constraints")
    print(f"Skipped {skipped_benign} benign samples")
    print(f"\nAverage lengths:")
    print(f"  - Prompt: {avg_prompt_len:.1f} chars")
    print(f"  - Chosen: {avg_chosen_len:.1f} chars")
    print(f"  - Rejected: {avg_rejected_len:.1f} chars")
    
    return output_file


def download_alpaca_benign_data(output_dir="data/benign", max_prompt_len=200, max_response_len=300,
                                 target_samples=10000):
    """
    Download BENIGN samples from tatsu-lab/alpaca dataset for training.
    Prioritizes shorter prompts and responses suitable for small LMs.
    
    For benign prompts:
    - DPO: chosen=helpful response, rejected=over_refusal
    """
    print(f"Downloading Alpaca BENIGN training data to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Alpaca dataset
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    candidates = []
    filtered_count = 0
    
    print(f"Processing Alpaca samples (benign only, prioritizing shortest prompts/responses)...")
    print(f"  - Max prompt length: {max_prompt_len} chars")
    print(f"  - Max response length: {max_response_len} chars")
    print(f"  - Target samples: {target_samples}")
    
    for item in tqdm(dataset):
        # Alpaca format: instruction + input (optional) + output
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        # Combine instruction and input to form the prompt
        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction
        
        if not prompt or not output:
            continue
        
        # Filter for short prompts and responses
        if len(prompt) > max_prompt_len or len(output) > max_response_len:
            filtered_count += 1
            continue
        
        # For benign prompts: chosen=helpful response, rejected=over-refusal
        chosen_response = output.strip()
        rejected_response = generate_over_refusal(prompt)
        
        # Store with total length for sorting
        total_len = len(prompt) + len(output)
        candidates.append({
            "prompt": prompt.strip(),
            "chosen": chosen_response,
            "rejected": rejected_response,
            "type": "benign",
            "_total_len": total_len
        })
        
        # Collect more than needed, then sort and filter
        if len(candidates) >= target_samples * 3:
            break
    
    # Remove duplicates based on prompt (keep first occurrence)
    seen_prompts = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate["prompt"] not in seen_prompts:
            seen_prompts.add(candidate["prompt"])
            unique_candidates.append(candidate)
    
    duplicates_removed = len(candidates) - len(unique_candidates)
    print(f"Removed {duplicates_removed} duplicate prompts")
    
    # Sort by total length (shortest first) and take the target number
    unique_candidates.sort(key=lambda x: x["_total_len"])
    samples = unique_candidates[:target_samples]
    
    # Remove the temporary _total_len field
    for sample in samples:
        del sample["_total_len"]
    
    output_file = os.path.join(output_dir, "benign_samples.json")
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    # Calculate average lengths
    avg_prompt_len = sum(len(s["prompt"]) for s in samples) / len(samples)
    avg_chosen_len = sum(len(s["chosen"]) for s in samples) / len(samples)
    avg_rejected_len = sum(len(s["rejected"]) for s in samples) / len(samples)
    
    print(f"\nDownloaded {len(samples)} benign samples to {output_file}")
    print(f"Filtered out {filtered_count} samples due to length constraints")
    print(f"\nAverage lengths:")
    print(f"  - Prompt: {avg_prompt_len:.1f} chars")
    print(f"  - Chosen: {avg_chosen_len:.1f} chars")
    print(f"  - Rejected: {avg_rejected_len:.1f} chars")
    
    return output_file


def download_benign_test_benchmark(output_dir="data/benchmarks", max_prompt_len=300, max_response_len=400,
                                    target_samples=500):
    """
    Download BENIGN test samples from Alpaca for evaluation.
    These are used to test if the model over-refuses safe prompts.
    
    Args:
        output_dir: Directory to save benchmark data
        max_prompt_len: Maximum prompt length
        max_response_len: Maximum response length
        target_samples: Number of benign test samples to collect
    """
    print(f"Downloading Alpaca BENIGN test benchmark to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load Alpaca dataset - use a different offset to avoid training data overlap
    dataset = load_dataset("tatsu-lab/alpaca", split="train")
    
    candidates = []
    filtered_count = 0
    start_offset = 40000  # Start from a different part of the dataset to avoid overlap
    
    print(f"Processing Alpaca samples for benign test set...")
    print(f"  - Max prompt length: {max_prompt_len} chars")
    print(f"  - Max response length: {max_response_len} chars")
    print(f"  - Target samples: {target_samples}")
    print(f"  - Starting from offset {start_offset} to avoid training overlap")
    
    for idx, item in enumerate(tqdm(dataset)):
        # Skip first 40K samples to ensure no overlap with training data
        if idx < start_offset:
            continue
            
        # Alpaca format: instruction + input (optional) + output
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        # Combine instruction and input to form the prompt
        if input_text:
            prompt = f"{instruction}\n{input_text}"
        else:
            prompt = instruction
        
        if not prompt or not output:
            continue
        
        # Filter for short prompts and responses
        if len(prompt) > max_prompt_len or len(output) > max_response_len:
            filtered_count += 1
            continue
        
        # Store with total length for sorting
        total_len = len(prompt) + len(output)
        candidates.append({
            "prompt": prompt.strip(),
            "target": output.strip(),  # Expected helpful response
            "type": "benign",
            "_len": total_len
        })
        
        # Collect more than needed, then sort and filter
        if len(candidates) >= target_samples * 2:
            break
    
    # Remove duplicates based on prompt
    seen_prompts = set()
    unique_candidates = []
    for candidate in candidates:
        if candidate["prompt"] not in seen_prompts:
            seen_prompts.add(candidate["prompt"])
            unique_candidates.append(candidate)
    
    duplicates_removed = len(candidates) - len(unique_candidates)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate prompts")
    
    # Sort by length (shortest first) and take target number
    unique_candidates.sort(key=lambda x: x["_len"])
    samples = unique_candidates[:target_samples]
    
    # Remove temporary _len field
    for sample in samples:
        del sample["_len"]
    
    output_file = os.path.join(output_dir, "benign_test.json")
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    avg_prompt_len = sum(len(s["prompt"]) for s in samples) / len(samples) if samples else 0
    avg_target_len = sum(len(s["target"]) for s in samples) / len(samples) if samples else 0
    
    print(f"\nDownloaded {len(samples)} benign test samples to {output_file}")
    print(f"Filtered out {filtered_count} samples due to length constraints")
    print(f"Average prompt length: {avg_prompt_len:.1f} chars")
    print(f"Average target length: {avg_target_len:.1f} chars")
    
    return output_file


def download_jailbreak_benchmark(output_dir="data/benchmarks", max_prompt_len=150):
    """
    Download JailbreakBench dataset for benchmarking.
    Prioritizes shorter prompts suitable for small LMs.
    """
    print(f"\nDownloading JailbreakBench benchmark to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    import jailbreakbench as jbb
    
    # Load full JBB behaviors dataset
    jbb_dataset = jbb.read_dataset()
    
    candidates = []
    filtered_count = 0
    
    # Access the behaviors, goals, and targets from the dataset
    behaviors = jbb_dataset.behaviors
    goals = jbb_dataset.goals
    targets = jbb_dataset.targets
    
    print(f"Processing {len(behaviors)} JailbreakBench behaviors...")
    
    # Combine them into our format
    for i, behavior in enumerate(behaviors):
        goal = goals[i] if i < len(goals) else ""
        target = targets[i] if i < len(targets) else ""
        
        # Filter for short prompts
        if len(goal) > max_prompt_len:
            filtered_count += 1
            continue
        
        candidates.append({
            "prompt": goal,
            "target": target,
            "behavior": behavior,
            "type": "adversarial",
            "_len": len(goal)
        })
    
    # Sort by length (shortest first)
    candidates.sort(key=lambda x: x["_len"])
    
    # Remove temporary _len field
    samples = [{k: v for k, v in c.items() if k != "_len"} for c in candidates]
    
    output_file = os.path.join(output_dir, "jailbreakbench_test.json")
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    avg_prompt_len = sum(len(s["prompt"]) for s in samples) / len(samples) if samples else 0
    
    print(f"Downloaded {len(samples)} JailbreakBench samples to {output_file}")
    print(f"Filtered out {filtered_count} samples due to length constraints")
    print(f"Average prompt length: {avg_prompt_len:.1f} chars")
    
    return output_file


def download_advbench_benchmark(output_dir="data/benchmarks", max_prompt_len=150):
    """
    Download AdvBench dataset for benchmarking.
    Prioritizes shorter prompts suitable for small LMs.
    """
    print(f"\nDownloading AdvBench benchmark to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load AdvBench harmful behaviors dataset with authentication
    dataset = load_dataset("walledai/AdvBench", split="train")
    
    candidates = []
    filtered_count = 0
    
    print(f"Processing {len(dataset)} AdvBench samples...")
    
    for item in dataset:
        prompt = item.get("prompt", item.get("goal", ""))
        target = item.get("target", "")
        
        # Filter for short prompts
        if len(prompt) > max_prompt_len:
            filtered_count += 1
            continue
        
        candidates.append({
            "prompt": prompt,
            "target": target,
            "behavior": item.get("behavior", "harmful"),
            "type": "adversarial",
            "_len": len(prompt)
        })
    
    # Sort by length (shortest first)
    candidates.sort(key=lambda x: x["_len"])
    
    # Remove temporary _len field
    samples = [{k: v for k, v in c.items() if k != "_len"} for c in candidates]
    
    output_file = os.path.join(output_dir, "advbench_test.json")
    with open(output_file, 'w') as f:
        json.dump(samples, f, indent=2)
    
    avg_prompt_len = sum(len(s["prompt"]) for s in samples) / len(samples) if samples else 0
    
    print(f"Downloaded {len(samples)} AdvBench samples to {output_file}")
    print(f"Filtered out {filtered_count} samples due to length constraints")
    print(f"Average prompt length: {avg_prompt_len:.1f} chars")
    
    return output_file


if __name__ == "__main__":
    print("=" * 60)
    print("Downloading Datasets for Prompt Injection Defense Project")
    print("=" * 60)
    print("\nConfiguration (optimized for small LMs):")
    print("  - Max prompt length: 300 characters")
    print("  - Max response length: 400 characters")
    print("  - Target training samples: 10K adversarial + 10K benign")
    print("  - Prioritizing shortest prompts/responses")
    print("  - Multiple varied rejected responses")
    print("\nTraining Data Sources:")
    print("  - Adversarial: WildJailbreak (harmful prompts with refusals)")
    print("  - Benign: Alpaca (safe prompts with helpful responses)")
    print("\nTesting Data Sources:")
    print("  - JailbreakBench (adversarial benchmark)")
    print("  - AdvBench (adversarial benchmark)")
    print("  - Benign test set: Alpaca (500 safe prompts)")
    print("=" * 60)
    
    # Download training data - adversarial from WildJailbreak
    print("\n[1/5] Downloading adversarial training samples from WildJailbreak...")
    adversarial_file = download_wildjailbreak_data(
        max_prompt_len=300,
        max_response_len=400,
        target_samples=10000
    )
    
    # Download training data - benign from Alpaca
    print("\n[2/5] Downloading benign training samples from Alpaca...")
    benign_file = download_alpaca_benign_data(
        max_prompt_len=300,
        max_response_len=400,
        target_samples=10000
    )
    
    # Download benchmark datasets for testing
    print("\n[3/5] Downloading JailbreakBench test dataset...")
    jailbreak_file = download_jailbreak_benchmark(max_prompt_len=300)
    
    print("\n[4/5] Downloading AdvBench test dataset...")
    advbench_file = download_advbench_benchmark(max_prompt_len=300)
    
    print("\n[5/5] Downloading Benign test benchmark from Alpaca...")
    benign_test_file = download_benign_test_benchmark(
        max_prompt_len=300,
        max_response_len=400,
        target_samples=500
    )
    
    print("\n" + "=" * 60)
    print("Data download complete!")
    print("\nTraining data (for SFT/DPO fine-tuning):")
    print(f"  - Adversarial: {adversarial_file}")
    print(f"  - Benign: {benign_file}")
    print("\nTesting data (for evaluation):")
    print(f"  - JailbreakBench (adversarial): {jailbreak_file}")
    print(f"  - AdvBench (adversarial): {advbench_file}")
    print(f"  - Benign test set: {benign_test_file}")
    print("\n💡 Use adversarial benchmarks to test refusal capability")
    print("💡 Use benign test set to test for over-refusal")
    print("=" * 60)
