# Evaluation System Guide

This guide explains how to use the updated evaluation system for testing models on adversarial and benign benchmarks.

## Overview

The evaluation system has been streamlined to:
- **Use Tinker only** (no local model evaluation)
- **Use LLM judge only** (no heuristic-based evaluation)
- **Separate judge prompts** for adversarial vs benign benchmarks
- Support multiple adversarial benchmark files

## Benchmark Files

### Adversarial Benchmarks
- `data/benchmarks/advbench_test.json` - AdvBench adversarial prompts
- `data/benchmarks/jailbreakbench_test.json` - JailbreakBench adversarial prompts

### Benign Benchmark
- `data/benchmarks/benign_test.json` - Benign/safe prompts

## Scripts

### 1. Evaluate Baseline Model

Evaluate a base model (before fine-tuning) to establish baseline metrics.

```bash
python evaluate_baseline.py \
  --base_model "meta-llama/Llama-3.2-1B" \
  --adversarial_files data/benchmarks/advbench_test.json data/benchmarks/jailbreakbench_test.json \
  --benign_file data/benchmarks/benign_test.json \
  --output_file results/baseline_evaluation.json \
  --judge_model "Qwen/Qwen3-235B-A22B-Instruct-2507" \
  --max_samples 100
```

**Key Arguments:**
- `--base_model`: HuggingFace model name to evaluate
- `--adversarial_files`: List of adversarial benchmark files
- `--benign_file`: Benign benchmark file
- `--judge_model`: LLM judge model (default: Qwen3-235B)
- `--max_samples`: Maximum samples to evaluate per benchmark (default: 100)

### 2. Evaluate Fine-tuned Model

Evaluate a fine-tuned model using its Tinker model ID.

```bash
python evaluate_sft.py \
  --model_id "tinker://YOUR_MODEL_ID_HERE" \
  --adversarial_files data/benchmarks/advbench_test.json data/benchmarks/jailbreakbench_test.json \
  --benign_file data/benchmarks/benign_test.json \
  --output_file results/sft_evaluation.json \
  --judge_model "Qwen/Qwen3-235B-A22B-Instruct-2507" \
  --base_model_name "meta-llama/Llama-3.2-1B" \
  --max_samples 100
```

**Key Arguments:**
- `--model_id`: Tinker model ID (e.g., `tinker://...`)
- `--base_model_name`: Base model name for tokenizer (should match training base model)
- Other arguments same as baseline evaluation

### 3. General Evaluation (Advanced)

Use the general evaluation script for custom configurations.

```bash
python evaluate.py \
  --model_path "meta-llama/Llama-3.2-1B" \
  --adversarial_files data/benchmarks/advbench_test.json data/benchmarks/jailbreakbench_test.json \
  --benign_file data/benchmarks/benign_test.json \
  --output_file results/custom_evaluation.json \
  --judge_model "Qwen/Qwen3-235B-A22B-Instruct-2507" \
  --max_samples 100
```

## LLM Judge

The evaluation system uses different judge prompts for different benchmarks:

### Adversarial Prompt Evaluation
- Uses `LLMJudge.judge_jailbreak_response()`
- Determines if the model was "jailbroken" (complied with harmful request)
- Checks if the model properly refused the request
- Returns: `is_jailbroken`, `is_refusal`, `confidence`, `explanation`

### Benign Prompt Evaluation
- Uses `LLMJudge.judge_benign_over_refusal()`
- Determines if the model over-refused a safe request
- Checks if the model provided a helpful response
- Returns: `is_over_refusal`, `is_helpful`, `explanation`

## Metrics

The evaluation produces the following metrics:

### Adversarial Robustness
- **Attack Success Rate (ASR)**: % of adversarial prompts that succeeded (lower is better)
- **Refusal Rate**: % of adversarial prompts properly refused (higher is better)

### Benign Helpfulness
- **Over-refusal Rate**: % of benign prompts incorrectly refused (lower is better)
- **Helpful Rate**: % of benign prompts answered helpfully (higher is better)
- **Average Response Length**: Average character count of responses

## Output Format

Results are saved in JSON format:

```json
{
  "model_path": "meta-llama/Llama-3.2-1B",
  "judge_model": "Qwen/Qwen3-235B-A22B-Instruct-2507",
  "adversarial_robustness": {
    "asr": 45.2,
    "refusal_rate": 54.8,
    "attack_successes": 45,
    "total_attacks": 100
  },
  "benign_helpfulness": {
    "over_refusal_rate": 12.3,
    "helpful_rate": 87.7,
    "over_refusals": 12,
    "helpful_responses": 88,
    "total_benign": 100,
    "avg_response_length": 145.6
  },
  "detailed_results": {
    "adversarial": [...],
    "benign": [...]
  }
}
```

## Requirements

- **Tinker API Key**: Set as environment variable
  ```bash
  export TINKER_API_KEY=your_key_here
  ```

- **Dependencies**:
  - `tinker`
  - `transformers`
  - Standard Python libraries (json, os, argparse)

## Tips

1. **Start with baseline**: Always run baseline evaluation first to establish comparison metrics
2. **Use same judge model**: For fair comparison, use the same judge model for all evaluations
3. **Consistent sample size**: Use the same `--max_samples` value for all evaluations
4. **Check API limits**: LLM judge makes many API calls; monitor usage and costs

## Example Workflow

```bash
# Step 1: Evaluate baseline model
python evaluate_baseline.py --max_samples 50

# Step 2: Train your model (using your training script)
# python train_sft.py ...

# Step 3: Evaluate fine-tuned model
python evaluate_sft.py \
  --model_id "tinker://YOUR_TRAINED_MODEL_ID" \
  --max_samples 50

# Step 4: Compare results
# Compare results/baseline_evaluation.json with results/sft_evaluation.json
```

## Troubleshooting

### "TINKER_API_KEY not found"
Set the environment variable:
```bash
export TINKER_API_KEY=your_key_here
```

### "Benchmark not found"
Ensure benchmark files exist at the specified paths. Check that you've downloaded the data.

### LLM Judge Errors
- Verify your Tinker API key has access to the judge model
- Try a different judge model if the specified one is unavailable
- Check API rate limits and quotas

