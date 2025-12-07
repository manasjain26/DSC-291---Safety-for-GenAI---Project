"""
Evaluate baseline (unmodified) Llama-3-8B model on jailbreak prompts.
This establishes the baseline Attack Success Rate before any fine-tuning.
"""

import os
import sys
import argparse

# Simply use the existing evaluate.py with the base model
from evaluate import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate baseline Llama-3-8B on jailbreak prompts"
    )
    parser.add_argument("--base_model", type=str, 
                        default="meta-llama/Llama-3.2-1B",
                        help="Base model to evaluate (ungated models: Mistral, Qwen, Llama-3.2)")
    parser.add_argument("--jailbreak_file", type=str,
                        default="data/jailbreak/jailbreak_test.json",
                        help="Path to jailbreak test data")
    parser.add_argument("--benign_file", type=str,
                        default="data/benign/benign_test.json",
                        help="Path to benign test data")
    parser.add_argument("--output_file", type=str,
                        default="results/baseline_evaluation_test.json",
                        help="Output file for baseline results")
    parser.add_argument("--use_llm_judge", action="store_true",
                        help="Use LLM judge for more accurate evaluation (recommended)")
    parser.add_argument("--judge_model", type=str,
                        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
                        help="Model to use as judge (e.g., Qwen/Qwen2.5-72B-Instruct)")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BASELINE EVALUATION - Original Llama-3-1B (No Fine-tuning)")
    print("="*80)
    print(f"\nModel: {args.base_model}")
    print("This will establish baseline ASR before any training.")
    print("\nNote: This may take 10-15 minutes depending on your GPU.")
    print("="*80 + "\n")
    
    # Check if data exists
    if not os.path.exists(args.jailbreak_file):
        print(f"❌ Jailbreak data not found: {args.jailbreak_file}")
        print("Please run: python download_data.py")
        sys.exit(1)
    
    if not os.path.exists(args.benign_file):
        print(f"❌ Benign data not found: {args.benign_file}")
        print("Please run: python download_data.py")
        sys.exit(1)
    
    # Run evaluation
    results = run_evaluation(
        model_path=args.base_model,
        jailbreak_file=args.jailbreak_file,
        benign_file=args.benign_file,
        output_file=args.output_file,
        use_tinker=True,
        use_llm_judge=args.use_llm_judge,
        judge_model=args.judge_model
    )
    
    print("\n" + "="*80)
    print("BASELINE RESULTS")
    print("="*80)
    print(f"\n📊 Attack Success Rate (ASR): {results['jailbreak_robustness']['asr']:.2f}%")
    print(f"   (This is the percentage of jailbreak prompts that succeeded)")
    
    # Check if benign evaluation completed
    if 'benign_helpfulness' in results:
        print(f"\n📊 Over-refusal Rate: {results['benign_helpfulness']['over_refusal_rate']:.2f}%")
        print(f"   (This is the percentage of benign prompts incorrectly refused)")
    else:
        print(f"\n⚠️  Benign helpfulness evaluation did not complete")
    
    print("\nThese metrics will serve as your baseline for comparison.")
    print("="*80 + "\n")

