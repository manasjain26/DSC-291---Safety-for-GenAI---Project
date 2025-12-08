"""
Evaluate baseline (unmodified) model on adversarial and benign benchmarks.
This establishes the baseline metrics before any fine-tuning.
"""

import os
import sys
import argparse
from evaluate import run_evaluation

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate baseline model on benchmarks"
    )
    parser.add_argument("--base_model", type=str, 
                        default="meta-llama/Llama-3.2-3B",
                        help="Base model to evaluate")
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
                        default="results/baseline_evaluation.json",
                        help="Output file for baseline results")
    parser.add_argument("--judge_model", type=str,
                        default="deepseek-ai/DeepSeek-V3.1",
                        help="LLM judge model")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum samples to evaluate per benchmark")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("BASELINE EVALUATION - Base Model (No Fine-tuning)")
    print("="*80)
    print(f"\nModel: {args.base_model}")
    print(f"Judge: {args.judge_model}")
    print("This will establish baseline metrics before training.")
    print("="*80 + "\n")
    
    # Check if data exists
    for adv_file in args.adversarial_files:
        if not os.path.exists(adv_file):
            print(f"❌ Adversarial benchmark not found: {adv_file}")
            sys.exit(1)
    
    if not os.path.exists(args.benign_file):
        print(f"❌ Benign benchmark not found: {args.benign_file}")
        sys.exit(1)
    
    # Check Tinker API key
    if not os.getenv("TINKER_API_KEY"):
        print("❌ ERROR: TINKER_API_KEY not found!")
        print("Please set your API key: export TINKER_API_KEY=your_key_here")
        sys.exit(1)
    
    # Run evaluation
    results = run_evaluation(
        model_path=args.base_model,
        adversarial_files=args.adversarial_files,
        benign_file=args.benign_file,
        output_file=args.output_file,
        judge_model=args.judge_model,
        max_samples=args.max_samples,
        base_model_name=args.base_model
    )
    
    print("\n" + "="*80)
    print("BASELINE RESULTS")
    print("="*80)
    print(f"\n📊 Adversarial Robustness:")
    print(f"   Attack Success Rate (ASR): {results['adversarial_robustness']['asr']:.2f}% (jailbroken - WORST)")
    print(f"   Refusal Rate: {results['adversarial_robustness']['refusal_rate']:.2f}% (proper refusals - BEST)")
    print(f"   Evasive Rate: {results['adversarial_robustness']['evasive_rate']:.2f}% (gibberish/non-answers - NEUTRAL)")
    
    print(f"\n📊 Benign Helpfulness:")
    print(f"   Helpful Rate: {results['benign_helpfulness']['helpful_rate']:.2f}% (BEST)")
    print(f"   Over-refusal Rate: {results['benign_helpfulness']['over_refusal_rate']:.2f}% (WORST)")
    
    print("\n💡 Interpretation:")
    print("   - ASR: Lower is better (fewer successful jailbreaks)")
    print("   - Refusal Rate: Higher is better (more proper refusals)")
    print("   - Evasive Rate: Model outputs gibberish instead of clear refusal or compliance")
    print("   - Helpful Rate: Higher is better (more benign prompts answered)")
    print("   - Over-refusal Rate: Lower is better (fewer safe prompts refused)")
    
    print("\nThese metrics will serve as your baseline for comparison.")
    print("="*80 + "\n")

