"""
Evaluate the SFT-trained model saved on Tinker servers.
Measures improvement over baseline on adversarial robustness and benign helpfulness.
Can optionally skip LLM judge and only generate responses.
"""

import os
import sys
import json
import argparse
import tinker
from transformers import AutoTokenizer
from tqdm import tqdm
from evaluate import run_evaluation


def load_eval_data(adversarial_files, benign_file):
    """Load evaluation datasets."""
    adversarial_data = []
    for file_path in adversarial_files:
        with open(file_path, 'r') as f:
            data = json.load(f)
            adversarial_data.extend(data)
    
    with open(benign_file, 'r') as f:
        benign_data = json.load(f)
    
    return adversarial_data, benign_data


def generate_response(model_client, tokenizer, prompt, max_tokens=256):
    """Generate response using Tinker sampling client."""
    try:
        # Encode the prompt into a ModelInput object
        encoded_prompt = tokenizer.encode(prompt, add_special_tokens=False)
        model_input = tinker.types.ModelInput.from_ints(encoded_prompt)
        
        # Define sampling parameters
        sampling_params = tinker.types.SamplingParams(
            max_tokens=max_tokens,
            temperature=0.0,  # Greedy decoding for reproducibility
            stop=["\n\n"]  # Stop at double newline
        )
        
        # Generate samples
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


def generate_responses_only(
    model_path,
    adversarial_files,
    benign_file,
    output_file,
    max_samples,
    base_model_name
):
    """
    Generate model responses without LLM judge evaluation.
    Only saves the raw responses for later analysis.
    """
    print("=" * 80)
    print("Generating Model Responses (No LLM Judge)")
    print("=" * 80)
    print(f"Model: {model_path}")
    print(f"Max samples per benchmark: {max_samples}")
    
    # Check Tinker API key
    if not os.getenv("TINKER_API_KEY"):
        raise ValueError("TINKER_API_KEY environment variable is required")
    
    # Load evaluation data
    adversarial_data, benign_data = load_eval_data(adversarial_files, benign_file)
    print(f"\nLoaded {len(adversarial_data)} adversarial samples")
    print(f"Loaded {len(benign_data)} benign samples")
    
    # Initialize Tinker
    print("\nInitializing Tinker...")
    service_client = tinker.ServiceClient()
    
    # Check if model_path is a Tinker model ID or base model
    is_tinker_model_id = model_path.startswith("tinker://")
    
    if is_tinker_model_id:
        print(f"Loading trained model from Tinker: {model_path}")
        model_client = service_client.create_sampling_client(model_path=model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    else:
        print(f"Loading base model: {model_path}")
        model_client = service_client.create_sampling_client(base_model=model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("✅ Model loaded")
    
    # Generate adversarial responses
    print("\n" + "=" * 60)
    print("Generating Adversarial Responses")
    print("=" * 60)
    adversarial_responses = []
    for item in tqdm(adversarial_data[:max_samples], 
                     desc="Adversarial", 
                     unit="prompt",
                     ncols=100):
        prompt = item["prompt"]
        response = generate_response(model_client, tokenizer, prompt)
        adversarial_responses.append({
            "prompt": prompt,
            "response": response
        })
    
    # Generate benign responses
    print("\n" + "=" * 60)
    print("Generating Benign Responses")
    print("=" * 60)
    benign_responses = []
    for item in tqdm(benign_data[:max_samples], 
                     desc="Benign", 
                     unit="prompt",
                     ncols=100):
        prompt = item["prompt"]
        response = generate_response(model_client, tokenizer, prompt)
        benign_responses.append({
            "prompt": prompt,
            "response": response
        })
    
    # Save responses
    responses_data = {
        "model_path": model_path,
        "adversarial_responses": adversarial_responses,
        "benign_responses": benign_responses
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(responses_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Response Generation Complete!")
    print("=" * 80)
    print(f"\n📁 Responses saved to: {output_file}")
    print(f"   - {len(adversarial_responses)} adversarial responses")
    print(f"   - {len(benign_responses)} benign responses")
    print("=" * 80 + "\n")
    
    return responses_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate SFT-trained model from Tinker"
    )
    parser.add_argument("--model_id", type=str, required=True,
                        help="Tinker model ID (e.g., tinker://...)")
    parser.add_argument("--adversarial_files", type=str, nargs="+",
                        default=[
                            "data/benchmarks/advbench_test.json",
                            "data/benchmarks/jailbreakbench_test.json"
                        ],
                        help="Paths to adversarial benchmark files")
    parser.add_argument("--benign_file", type=str,
                        default="data/benchmarks/benign_test.json",
                        help="Path to benign benchmark file")
    parser.add_argument("--output_file", type=str,
                        default="results/sft_evaluation.json",
                        help="Output file for results")
    parser.add_argument("--judge_model", type=str,
                        default="deepseek-ai/DeepSeek-V3.1",
                        help="LLM judge model")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum samples to evaluate per benchmark")
    parser.add_argument("--base_model_name", type=str,
                        default="meta-llama/Llama-3.2-3B",
                        help="Base model name for tokenizer")
    parser.add_argument("--responses_only", action="store_true",
                        help="Only generate responses without LLM judge evaluation")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SFT MODEL EVALUATION")
    print("="*80)
    print(f"\nTinker Model ID: {args.model_id}")
    if not args.responses_only:
        print(f"Judge Model: {args.judge_model}")
    else:
        print("Mode: Responses Only (No LLM Judge)")
    print("Evaluating on benchmarks...")
    print("="*80 + "\n")
    
    # Check API key
    if not os.getenv("TINKER_API_KEY"):
        print("❌ ERROR: TINKER_API_KEY not found!")
        print("Please set your API key: export TINKER_API_KEY=your_key_here")
        sys.exit(1)
    
    # Check if benchmark data exists
    for adv_file in args.adversarial_files:
        if not os.path.exists(adv_file):
            print(f"❌ Adversarial benchmark not found: {adv_file}")
            sys.exit(1)
    
    if not os.path.exists(args.benign_file):
        print(f"❌ Benign benchmark not found: {args.benign_file}")
        sys.exit(1)
    
    # Branch based on responses_only flag
    if args.responses_only:
        # Only generate responses without LLM judge
        results = generate_responses_only(
            model_path=args.model_id,
            adversarial_files=args.adversarial_files,
            benign_file=args.benign_file,
            output_file=args.output_file,
            max_samples=args.max_samples,
            base_model_name=args.base_model_name
        )
    else:
        # Run full evaluation with LLM judge
        results = run_evaluation(
            model_path=args.model_id,
            adversarial_files=args.adversarial_files,
            benign_file=args.benign_file,
            output_file=args.output_file,
            judge_model=args.judge_model,
            max_samples=args.max_samples,
            base_model_name=args.base_model_name
        )
        
        print("\n" + "="*80)
        print("SFT MODEL RESULTS")
        print("="*80)
        print(f"\n📊 Adversarial Robustness:")
        print(f"   Attack Success Rate (ASR): {results['adversarial_robustness']['asr']:.2f}% (jailbroken - WORST)")
        print(f"   Refusal Rate: {results['adversarial_robustness']['refusal_rate']:.2f}% (proper refusals - BEST)")
        print(f"   Evasive Rate: {results['adversarial_robustness']['evasive_rate']:.2f}% (gibberish/non-answers - NEUTRAL)")
        
        print(f"\n📊 Benign Helpfulness:")
        print(f"   Helpful Rate: {results['benign_helpfulness']['helpful_rate']:.2f}% (BEST)")
        print(f"   Over-refusal Rate: {results['benign_helpfulness']['over_refusal_rate']:.2f}% (WORST)")
        
        print("\n💡 Compare with baseline to see improvement!")
        print("   Goal: Lower ASR, Higher Refusal Rate, Maintain Helpful Rate")
        
        print(f"\nResults saved to: {args.output_file}")
        print("="*80 + "\n")

