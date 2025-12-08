"""
Evaluate the SFT-trained model saved on Tinker servers.
Measures improvement over baseline on adversarial robustness and benign helpfulness.
"""

import os
import sys
import argparse
from evaluate import run_evaluation

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
                        default="Qwen/Qwen3-235B-A22B-Instruct-2507",
                        help="LLM judge model")
    parser.add_argument("--max_samples", type=int, default=1000,
                        help="Maximum samples to evaluate per benchmark")
    parser.add_argument("--base_model_name", type=str,
                        default="meta-llama/Llama-3.2-1B",
                        help="Base model name for tokenizer")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("SFT MODEL EVALUATION")
    print("="*80)
    print(f"\nTinker Model ID: {args.model_id}")
    print(f"Judge Model: {args.judge_model}")
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
    
    # Run evaluation
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

