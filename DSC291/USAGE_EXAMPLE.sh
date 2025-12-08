#!/bin/bash
# Example usage of the evaluation system
# This script shows how to evaluate baseline and fine-tuned models

# Set your Tinker API key
export TINKER_API_KEY="your_key_here"

# ============================================
# Step 1: Evaluate Baseline Model
# ============================================
echo "Step 1: Evaluating baseline model..."
python evaluate_baseline.py \
  --base_model "meta-llama/Llama-3.2-1B" \
  --adversarial_files data/benchmarks/advbench_test.json data/benchmarks/jailbreakbench_test.json \
  --benign_file data/benchmarks/benign_test.json \
  --output_file results/baseline_evaluation.json \
  --judge_model "Qwen/Qwen3-235B-A22B-Instruct-2507" \
  --max_samples 100

echo "✅ Baseline evaluation complete!"
echo "Results saved to: results/baseline_evaluation.json"

# ============================================
# Step 2: Train Your Model (example)
# ============================================
# This is where you would run your training script
# Example:
# python train_sft.py \
#   --base_model "meta-llama/Llama-3.2-1B" \
#   --train_data data/sft_train.json \
#   --output_dir models/sft_model

# ============================================
# Step 3: Evaluate Fine-tuned Model
# ============================================
echo ""
echo "Step 3: Evaluating fine-tuned model..."
# Replace with your actual Tinker model ID
MODEL_ID="tinker://YOUR_MODEL_ID_HERE"

python evaluate_sft.py \
  --model_id "$MODEL_ID" \
  --adversarial_files data/benchmarks/advbench_test.json data/benchmarks/jailbreakbench_test.json \
  --benign_file data/benchmarks/benign_test.json \
  --output_file results/sft_evaluation.json \
  --judge_model "Qwen/Qwen3-235B-A22B-Instruct-2507" \
  --base_model_name "meta-llama/Llama-3.2-1B" \
  --max_samples 100

echo "✅ SFT evaluation complete!"
echo "Results saved to: results/sft_evaluation.json"

# ============================================
# Step 4: Compare Results
# ============================================
echo ""
echo "Step 4: Comparing results..."
echo ""
echo "=== BASELINE MODEL ==="
cat results/baseline_evaluation.json | python -m json.tool | grep -A 5 "adversarial_robustness"
cat results/baseline_evaluation.json | python -m json.tool | grep -A 7 "benign_helpfulness"

echo ""
echo "=== FINE-TUNED MODEL ==="
cat results/sft_evaluation.json | python -m json.tool | grep -A 5 "adversarial_robustness"
cat results/sft_evaluation.json | python -m json.tool | grep -A 7 "benign_helpfulness"

echo ""
echo "✅ Evaluation complete! Check the results above."

