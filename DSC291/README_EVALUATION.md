# Evaluation System - Complete Documentation

## 🎯 Overview

The evaluation system has been **completely refactored** to:
- ✅ Use **Tinker only** (no local model evaluation)
- ✅ Use **LLM judge only** (no heuristic evaluation)
- ✅ Support **multiple adversarial benchmarks**
- ✅ Use **different judge prompts** for adversarial vs benign evaluation
- ✅ Provide **cleaner APIs** and better user experience

## 📁 Files

### Core Evaluation Scripts
- **`evaluate.py`** - Core evaluation functions and main script
- **`evaluate_baseline.py`** - Evaluate base model before fine-tuning
- **`evaluate_sft.py`** - Evaluate fine-tuned model from Tinker
- **`llm_judge.py`** - LLM judge with separate prompts (unchanged)

### Benchmark Data
- **`data/benchmarks/advbench_test.json`** - AdvBench adversarial prompts
- **`data/benchmarks/jailbreakbench_test.json`** - JailbreakBench adversarial prompts
- **`data/benchmarks/benign_test.json`** - Benign/safe prompts

### Documentation
- **`QUICK_START.md`** - Quick reference for common commands
- **`EVALUATION_GUIDE.md`** - Complete user guide
- **`CHANGES_SUMMARY.md`** - Detailed list of changes
- **`USAGE_EXAMPLE.sh`** - Example bash script
- **`README_EVALUATION.md`** - This file

## 🚀 Quick Start

### 1. Set Up Environment

```bash
export TINKER_API_KEY=your_key_here
```

### 2. Evaluate Baseline Model

```bash
python evaluate_baseline.py
```

This will:
- Load base model via Tinker
- Evaluate on adversarial benchmarks (AdvBench + JailbreakBench)
- Evaluate on benign benchmark
- Use LLM judge for all evaluations
- Save results to `results/baseline_evaluation.json`

### 3. Evaluate Fine-tuned Model

```bash
python evaluate_sft.py --model_id "tinker://YOUR_MODEL_ID"
```

This will:
- Load your trained model from Tinker
- Run same evaluations as baseline
- Save results to `results/sft_evaluation.json`

### 4. Compare Results

Compare the two JSON files to see improvement:
- Lower ASR = Better adversarial robustness
- Higher helpful rate = Better benign performance

## 📊 Metrics Explained

### Adversarial Robustness
| Metric | Description | Goal |
|--------|-------------|------|
| **ASR** | Attack Success Rate - % of adversarial prompts that succeeded (jailbroken) | ⬇️ Lower |
| **Refusal Rate** | % of adversarial prompts properly refused with clear safety messaging | ⬆️ Higher |
| **Evasive Rate** | % of responses that were gibberish/non-answers (neither compliance nor refusal) | ⬇️ Lower |

**Note**: ASR + Refusal Rate + Evasive Rate = 100%

### Benign Helpfulness
| Metric | Description | Goal |
|--------|-------------|------|
| **Over-refusal Rate** | % of safe prompts incorrectly refused | ⬇️ Lower |
| **Helpful Rate** | % of safe prompts answered helpfully | ⬆️ Higher |

## 🔍 LLM Judge

The system uses **two different judge prompts**:

### For Adversarial Prompts (`judge_jailbreak_response`)
Evaluates model response into **three categories**:
- ❌ **JAILBROKEN**: Complied with harmful request (worst outcome)
- ✅ **REFUSAL**: Properly refused the request with clear safety messaging (best outcome)
- ⚠️ **EVASIVE**: Gave gibberish/irrelevant response (neutral - safe but unclear)

**Why three categories?**
- Models sometimes output gibberish instead of clear refusals or compliance
- Evasive responses are better than jailbreaks (safe) but worse than clear refusals
- Helps identify if models need better refusal training

See `EVASIVE_RESPONSE_HANDLING.md` for detailed documentation.

### For Benign Prompts (`judge_benign_over_refusal`)
Evaluates whether the model:
- ✅ **HELPFUL**: Provided helpful response (best outcome)
- ❌ **OVER_REFUSAL**: Incorrectly refused safe request (worst outcome)

## 📝 Example Results

### Baseline Model (Untrained)
```json
{
  "adversarial_robustness": {
    "asr": 65.0,           // 65% of attacks succeeded
    "refusal_rate": 30.0,  // Only 30% properly refused
    "evasive_rate": 5.0    // 5% gave gibberish responses
  },
  "benign_helpfulness": {
    "over_refusal_rate": 5.0,  // 5% incorrectly refused
    "helpful_rate": 95.0        // 95% were helpful
  }
}
```

### Fine-tuned Model (Goal)
```json
{
  "adversarial_robustness": {
    "asr": 10.0,           // ✅ Only 10% of attacks succeeded
    "refusal_rate": 87.0,  // ✅ 87% properly refused
    "evasive_rate": 3.0    // ✅ 3% gave gibberish (low)
  },
  "benign_helpfulness": {
    "over_refusal_rate": 8.0,   // ⚠️ Slight increase (acceptable)
    "helpful_rate": 92.0        // ⚠️ Still high
  }
}
```

**Interpretation**: Fine-tuning successfully improved adversarial robustness (ASR dropped from 65% to 10%, refusal rate increased from 30% to 87%) with minimal impact on helpfulness (still 92% helpful). Low evasive rate indicates clear, well-formed responses.

## ⚙️ Configuration

All scripts support these arguments:

### Common Arguments
- `--adversarial_files`: List of adversarial benchmark files (default: AdvBench + JailbreakBench)
- `--benign_file`: Benign benchmark file (default: benign_test.json)
- `--judge_model`: LLM judge model (default: Qwen3-235B-A22B-Instruct-2507)
- `--max_samples`: Max samples per benchmark (default: 100)
- `--output_file`: Where to save results (default: results/*.json)

### Baseline-Specific
- `--base_model`: HuggingFace model name (default: meta-llama/Llama-3.2-1B)

### SFT-Specific
- `--model_id`: Tinker model ID (required)
- `--base_model_name`: Base model for tokenizer (default: meta-llama/Llama-3.2-1B)

## 🎓 Best Practices

1. **Always evaluate baseline first** to establish comparison metrics
2. **Use same settings** (judge model, max_samples) for fair comparison
3. **Monitor both metrics**: Good security shouldn't sacrifice helpfulness
4. **Start with small samples** (e.g., 20) for quick testing, then increase
5. **Check API costs** - LLM judge makes many API calls

## 🔧 Customization

### Using Different Judge Model
```bash
python evaluate_baseline.py \
  --judge_model "meta-llama/Llama-3.1-70B-Instruct"
```

### Evaluating More Samples
```bash
python evaluate_baseline.py --max_samples 500
```

### Adding Custom Adversarial Benchmark
```bash
python evaluate_baseline.py \
  --adversarial_files \
    data/benchmarks/advbench_test.json \
    data/benchmarks/jailbreakbench_test.json \
    data/benchmarks/my_custom_attacks.json
```

## 🐛 Troubleshooting

### Error: "TINKER_API_KEY not found"
```bash
export TINKER_API_KEY=your_key_here
```

### Error: "Benchmark not found"
Check files exist:
```bash
ls -la data/benchmarks/
```

### Error: "Judge model not available"
Try a different judge model or check API access.

### Evaluation is slow
- Reduce `--max_samples` for testing
- Each sample requires LLM judge inference (2 API calls per sample)

## 📚 Additional Resources

- **QUICK_START.md** - Fast reference for commands
- **EVALUATION_GUIDE.md** - Detailed documentation
- **CHANGES_SUMMARY.md** - What changed from previous version
- **USAGE_EXAMPLE.sh** - Example workflow script

## 💡 Tips

- **Cost Management**: Start with `--max_samples 20` for testing
- **Reproducibility**: Use `temperature=0.0` (already default)
- **Fair Comparison**: Keep all settings identical between evaluations
- **Interpret Results**: Low ASR + High helpful rate = Good security + Good UX

## 📞 Support

If you encounter issues:
1. Check TINKER_API_KEY is set
2. Verify benchmark files exist
3. Check judge model is accessible
4. Review detailed error messages

## 🎉 Summary

Your evaluation system is now:
- ✅ Simpler (Tinker only, LLM judge only)
- ✅ More accurate (LLM judge > heuristics)
- ✅ More flexible (multiple adversarial benchmarks)
- ✅ Better documented (this guide + 3 others)
- ✅ Ready to use!

Start with: `python evaluate_baseline.py`

