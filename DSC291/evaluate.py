"""
Evaluation script to measure Attack Success Rate (ASR) and helpfulness.
Tests models against adversarial and benign benchmarks using Tinker and LLM judge.
"""

import os
import json
import argparse
from typing import List, Dict
import tinker
from transformers import AutoTokenizer
from llm_judge import LLMJudge
from tqdm import tqdm


def load_eval_data(adversarial_files: List[str], benign_file: str) -> tuple:
    """Load evaluation datasets."""
    adversarial_data = []
    for file_path in adversarial_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            adversarial_data.extend(data)
    
    with open(benign_file, 'r') as f:
        benign_data = json.load(f)
    
    return adversarial_data, benign_data


def generate_response(model_client, tokenizer, prompt: str, max_tokens: int = 256) -> str:
    """Generate response using Tinker sampling client."""
    try:
        # Encode the prompt into a ModelInput object
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        model_input = tinker.types.ModelInput.from_ints(encoded_prompt)
        
        # Define sampling parameters
        # Use temperature=0 for deterministic, reproducible benchmarking
        sampling_params = tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,  # Greedy decoding for reproducibility
            stop=["\n\n"]  # Stop at double newline
        )
        
        # Generate samples using keyword arguments (as per Tinker docs)
        response = model_client.sample(
            prompt=model_input,
            sampling_params=sampling_params,
            num_samples=1
        ).result()
        
        # Extract and decode the generated tokens
        generated_tokens = response.sequences[0].tokens
        generated_text = tokenizer.decode(generated_tokens)
        
        return generated_text
    except Exception as e:
        return f"[ERROR: {str(e)}]"


def evaluate_adversarial_robustness(
    model_client,
    tokenizer,
    adversarial_data: List[Dict],
    llm_judge: LLMJudge,
    max_samples: int = 100
) -> Dict:
    """
    Evaluate model robustness against adversarial prompts.
    Measures Attack Success Rate (ASR) using LLM judge.
    Categorizes responses as: jailbroken, refusal, or evasive (gibberish).
    """
    print("\n" + "=" * 60)
    print("Evaluating Adversarial Robustness (ASR)")
    print("=" * 60)
    
    results = []
    attack_successes = 0
    refusals = 0
    evasive_responses = 0
    total_samples = min(len(adversarial_data), max_samples)
    
    # Use tqdm for progress bar
    for item in tqdm(adversarial_data[:max_samples], 
                     desc="Adversarial Eval", 
                     unit="prompt",
                     ncols=100):
        prompt = item["prompt"]
        
        # Generate response using Tinker
        response = generate_response(model_client, tokenizer, prompt)
        
        # Evaluate using LLM judge for adversarial prompts
        judgment = llm_judge.judge_jailbreak_response(prompt, response)
        attack_success = judgment["is_jailbroken"]
        refused = judgment["is_refusal"]
        evasive = judgment["is_evasive"]
        response_type = judgment["response_type"]
        
        result_entry = {
            "prompt": prompt,
            "response": response,
            "attack_success": attack_success,
            "refused": refused,
            "evasive": evasive,
            "response_type": response_type,
            "judge_explanation": judgment["explanation"],
            "judge_confidence": judgment["confidence"]
        }
        
        if attack_success:
            attack_successes += 1
        if refused:
            refusals += 1
        if evasive:
            evasive_responses += 1
        
        results.append(result_entry)
    
    asr = (attack_successes / len(results)) * 100
    refusal_rate = (refusals / len(results)) * 100
    evasive_rate = (evasive_responses / len(results)) * 100
    
    print(f"\n📊 Attack Success Rate (ASR): {asr:.2f}% (jailbroken)")
    print(f"📊 Refusal Rate: {refusal_rate:.2f}% (proper refusals)")
    print(f"📊 Evasive Rate: {evasive_rate:.2f}% (gibberish/non-answers)")
    print(f"Total attacks evaluated: {len(results)}")
    
    return {
        "asr": asr,
        "refusal_rate": refusal_rate,
        "evasive_rate": evasive_rate,
        "attack_successes": attack_successes,
        "refusals": refusals,
        "evasive_responses": evasive_responses,
        "total_attacks": len(results),
        "results": results
    }


def evaluate_benign_helpfulness(
    model_client,
    tokenizer,
    benign_data: List[Dict],
    llm_judge: LLMJudge,
    max_samples: int = 100
) -> Dict:
    """
    Evaluate model helpfulness on benign prompts using LLM judge.
    Measures over-refusal rate.
    """
    print("\n" + "=" * 60)
    print("Evaluating Benign Helpfulness")
    print("=" * 60)
    
    results = []
    over_refusals = 0
    helpful_responses = 0
    total_samples = min(len(benign_data), max_samples)
    
    # Use tqdm for progress bar
    for item in tqdm(benign_data[:max_samples], 
                     desc="Benign Eval", 
                     unit="prompt",
                     ncols=100):
        prompt = item["prompt"]
        
        # Generate response using Tinker
        response = generate_response(model_client, tokenizer, prompt)
        
        # Evaluate using LLM judge for benign prompts
        judgment = llm_judge.judge_benign_over_refusal(prompt, response)
        over_refused = judgment["is_over_refusal"]
        is_helpful = judgment["is_helpful"]
        
        if over_refused:
            over_refusals += 1
        if is_helpful:
            helpful_responses += 1
        
        results.append({
            "prompt": prompt,
            "response": response,
            "over_refused": over_refused,
            "is_helpful": is_helpful,
            "judge_explanation": judgment["explanation"],
            "response_length": len(response)
        })
        prompt = item["prompt"]
        
        # Generate response using Tinker
        response = generate_response(model_client, tokenizer, prompt)
        
        # Evaluate using LLM judge for benign prompts
        judgment = llm_judge.judge_benign_over_refusal(prompt, response)
        over_refused = judgment["is_over_refusal"]
        is_helpful = judgment["is_helpful"]
        
        if over_refused:
            over_refusals += 1
        if is_helpful:
            helpful_responses += 1
        
        results.append({
            "prompt": prompt,
            "response": response,
            "over_refused": over_refused,
            "is_helpful": is_helpful,
            "judge_explanation": judgment["explanation"],
            "response_length": len(response)
        })
    
    over_refusal_rate = (over_refusals / len(results)) * 100
    helpful_rate = (helpful_responses / len(results)) * 100
    avg_response_length = sum(r["response_length"] for r in results) / len(results)
    
    print(f"\nOver-refusal Rate: {over_refusal_rate:.2f}%")
    print(f"Helpful Rate: {helpful_rate:.2f}%")
    print(f"Average Response Length: {avg_response_length:.1f} chars")
    print(f"Total benign prompts evaluated: {len(results)}")
    
    return {
        "over_refusal_rate": over_refusal_rate,
        "helpful_rate": helpful_rate,
        "over_refusals": over_refusals,
        "helpful_responses": helpful_responses,
        "total_benign": len(results),
        "avg_response_length": avg_response_length,
        "results": results
    }


def run_evaluation(
    model_path: str,
    adversarial_files: List[str],
    benign_file: str,
    output_file: str = "results/evaluation_results.json",
    judge_model: str = "Qwen/Qwen3-235B-A22B-Instruct-2507",
    max_samples: int = 100,
    base_model_name: str = "meta-llama/Llama-3.2-1B"
):
    """
    Run full evaluation pipeline using Tinker and LLM judge.
    
    Args:
        model_path: Tinker model ID or base model name
        adversarial_files: List of paths to adversarial benchmark files
        benign_file: Path to benign benchmark file
        output_file: Path to save results
        judge_model: LLM judge model name
        max_samples: Maximum samples to evaluate per benchmark
        base_model_name: Base model name for tokenizer (if using Tinker model ID)
    """
    print("=" * 80)
    print("Starting Evaluation with Tinker + LLM Judge")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Adversarial benchmarks: {adversarial_files}")
    print(f"Benign benchmark: {benign_file}")
    print(f"Judge model: {judge_model}")
    print(f"Max samples per benchmark: {max_samples}")
    
    # Check Tinker API key
    if not os.getenv("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY environment variable is required")
    
    # Load evaluation data
    adversarial_data, benign_data = load_eval_data(adversarial_files, benign_file)
    print(f"\nLoaded {len(adversarial_data)} adversarial samples")
    print(f"Loaded {len(benign_data)} benign samples")
    
    # Initialize LLM judge
    print(f"\n🔍 Initializing LLM Judge: {judge_model}")
    llm_judge = LLMJudge(judge_model=judge_model)
    
    # Initialize Tinker
    print("\nInitializing Tinker...")
    service_client = tinker.ServiceClient()
    
    # Check if model_path is a Tinker model ID or base model
    is_tinker_model_id = model_path.startswith("tinker://")
    
    if is_tinker_model_id:
        # This is a Tinker model ID (trained model)
        print(f"Loading trained model from Tinker: {model_path}")
        model_client = service_client.create_sampling_client(model_path=model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        # This is a base model from HuggingFace
        print(f"Loading base model: {model_path}")
        model_client = service_client.create_sampling_client(base_model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("✅ Model loaded")
    
    # Evaluate adversarial robustness
    adversarial_results = evaluate_adversarial_robustness(
        model_client, tokenizer, adversarial_data, llm_judge, max_samples
    )
    
    # Evaluate benign helpfulness
    benign_results = evaluate_benign_helpfulness(
        model_client, tokenizer, benign_data, llm_judge, max_samples
    )
    
    # Combine results
    final_results = {
        "model_path": model_path,
        "judge_model": judge_model,
        "adversarial_robustness": {
            "asr": adversarial_results["asr"],
            "refusal_rate": adversarial_results["refusal_rate"],
            "evasive_rate": adversarial_results["evasive_rate"],
            "attack_successes": adversarial_results["attack_successes"],
            "refusals": adversarial_results["refusals"],
            "evasive_responses": adversarial_results["evasive_responses"],
            "total_attacks": adversarial_results["total_attacks"]
        },
        "benign_helpfulness": {
            "over_refusal_rate": benign_results["over_refusal_rate"],
            "helpful_rate": benign_results["helpful_rate"],
            "over_refusals": benign_results["over_refusals"],
            "helpful_responses": benign_results["helpful_responses"],
            "total_benign": benign_results["total_benign"],
            "avg_response_length": benign_results["avg_response_length"]
        },
        "detailed_results": {
            "adversarial": adversarial_results["results"],
            "benign": benign_results["results"]
        }
    }
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Save model responses separately (detailed responses only)
    responses_file = output_file.replace('.json', '_responses.json')
    responses_only = {
        "model_path": model_path,
        "adversarial_responses": [
            {
                "prompt": r["prompt"],
                "response": r["response"],
                "response_type": r["response_type"]
            }
            for r in adversarial_results["results"]
        ],
        "benign_responses": [
            {
                "prompt": r["prompt"],
                "response": r["response"],
                "is_helpful": r["is_helpful"]
            }
            for r in benign_results["results"]
        ]
    }
    with open(responses_file, 'w') as f:
        json.dump(responses_only, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print(f"\n📊 Adversarial Robustness:")
    print(f"  Attack Success Rate (ASR): {adversarial_results['asr']:.2f}% (jailbroken - WORST)")
    print(f"  Refusal Rate: {adversarial_results['refusal_rate']:.2f}% (proper refusals - BEST)")
    print(f"  Evasive Rate: {adversarial_results['evasive_rate']:.2f}% (gibberish/non-answers - NEUTRAL)")
    print(f"\n📊 Benign Helpfulness:")
    print(f"  Helpful Rate: {benign_results['helpful_rate']:.2f}% (BEST)")
    print(f"  Over-refusal Rate: {benign_results['over_refusal_rate']:.2f}% (WORST)")
    print(f"\n📁 Files saved:")
    print(f"  - Full results: {output_file}")
    print(f"  - Model responses: {responses_file}")
    print("=" * 80)
    
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on adversarial and benign benchmarks")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Tinker model ID or base model name")
    parser.add_argument("--adversarial_files", type=str, nargs="+",
                        default=["data/benchmarks/advbench_test.json", "data/benchmarks/jailbreakbench_test.json"],
                        help="Paths to adversarial benchmark files")
    parser.add_argument("--benign_file", type=str,
                        default="data/benchmarks/benign_test.json",
                        help="Path to benign benchmark file")
    parser.add_argument("--output_file", type=str,
                        default="results/evaluation_results.json",
                        help="Output file for results")
    parser.add_argument("--judge_model", type=str,
                        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
                        help="LLM judge model name")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum samples to evaluate per benchmark")
    parser.add_argument("--base_model_name", type=str,
                        default="meta-llama/Llama-3.2-1B",
                        help="Base model name for tokenizer (if using Tinker model ID)")
    args = parser.parse_args()
    
    run_evaluation(
        model_path=args.model_path,
        adversarial_files=args.adversarial_files,
        benign_file=args.benign_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        max_samples=args.max_samples,
        base_model_name=args.base_model_name
    )

