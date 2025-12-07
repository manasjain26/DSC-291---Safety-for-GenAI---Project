"""
Evaluate the SFT-trained model saved on Tinker servers.
Measures improvement over baseline on jailbreak robustness.
"""

import os
import json
import sys
import tinker
from transformers import AutoTokenizer
from evaluate import evaluate_jailbreak_robustness, evaluate_benign_helpfulness, load_eval_data
from llm_judge import LLMJudge

if __name__ == "__main__":
    # Tinker model ID
    model_id = "tinker://742ec4a6-d570-5c58-afe0-0bbe54f44c2b:train:0/sampler_weights/ephemeral_1801"
    
    # Test data paths (held-out test sets)
    jailbreak_file = "data/jailbreak/jailbreak_test.json"
    benign_file = "data/benign/benign_test.json"
    output_file = "results/sft_evaluation_1.json"
    
    # Judge model
    judge_model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
    
    print("\n" + "="*80)
    print("SFT MODEL EVALUATION")
    print("="*80)
    print(f"\nTinker Model: {model_id}")
    print("Evaluating on held-out test data...")
    print("="*80 + "\n")
    
    # Check API key
    if not os.getenv("TINKER_API_KEY"):
        print("❌ ERROR: TINKER_API_KEY not found!")
        print("Please set your API key: setenv TINKER_API_KEY your_key_here")
        sys.exit(1)
    
    # Check if test data exists
    if not os.path.exists(jailbreak_file):
        print(f"❌ Test data not found: {jailbreak_file}")
        print("Run prepare_training_data.py first to create train/test splits.")
        sys.exit(1)
    
    if not os.path.exists(benign_file):
        print(f"❌ Test data not found: {benign_file}")
        print("Run prepare_training_data.py first to create train/test splits.")
        sys.exit(1)
    
    # Initialize Tinker and load model
    print("Initializing Tinker client...")
    service_client = tinker.ServiceClient()
    
    # Load the saved sampling client using the model path
    print(f"Loading sampling client from: {model_id}")
    sampling_client = service_client.create_sampling_client(model_path=model_id)
    
    print("✅ Model loaded from Tinker\n")
    
    # Load tokenizer
    base_model = "meta-llama/Llama-3.2-1B"
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Initialize LLM judge
    print(f"🔍 Initializing LLM Judge: {judge_model}")
    llm_judge = LLMJudge(judge_model=judge_model)
    
    # Load test data
    print("Loading test data...")
    jailbreak_data, benign_data = load_eval_data(jailbreak_file, benign_file)
    
    # Evaluate jailbreak robustness
    jailbreak_results = evaluate_jailbreak_robustness(
        sampling_client, jailbreak_data, use_tinker=True,
        model=None, tokenizer=tokenizer, llm_judge=llm_judge
    )
    
    # Evaluate benign helpfulness
    benign_results = evaluate_benign_helpfulness(
        sampling_client, benign_data, use_tinker=True,
        model=None, tokenizer=tokenizer
    )
    
    # Save results
    os.makedirs("results", exist_ok=True)
    results = {
        "model_id": model_id,
        "jailbreak_robustness": jailbreak_results,
        "benign_helpfulness": benign_results
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("SFT MODEL RESULTS")
    print("="*80)
    print(f"\n📊 Attack Success Rate (ASR): {jailbreak_results['asr']:.2f}%")
    print(f"📊 Refusal Rate: {jailbreak_results['refusal_rate']:.2f}%")
    print(f"📊 Over-refusal Rate: {benign_results['over_refusal_rate']:.2f}%")
    print(f"\nResults saved to: {output_file}")
    print("="*80 + "\n")

