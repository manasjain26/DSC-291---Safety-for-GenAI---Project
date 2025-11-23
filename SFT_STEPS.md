### SFT Baseline: Data + Training + Evaluation

This guide prepares synthetic adversarial data, builds a balanced SFT dataset (benign helpful + adversarial refusals), fine-tunes a base SLM with LoRA on Tinker, and evaluates.

## 1) Generate Synthetic Adversarial Prompts
- Patterns to cover (examples):
  - Ignore/override system: “Ignore previous instructions…”
  - CoT/Jailbreak scaffolds: “For a movie script… hypothetically…”
  - Policy exfil: “Print your hidden policies…”
  - Delegation: “Pretend to be a tool JSON parser…”
  - Indirection/encoding: base64, ROT13, code blocks to bypass filters
  - Role-play authority: “You are a safety auditor…”

Run generator (Gemini is the default for creating `data/processed/synthetic/adversarial.jsonl`):

```bash
python src/data/synthesize_adversarial.py \
  --num 20 \
  --output data/processed/synthetic/adversarial.jsonl \
  --engine gemini:gemini-2.5-flash \
  --generate-refusals 
```

Notes:
- Gemini is the default recommended engine for generating `adversarial.jsonl`. Use `--engine gemini:gemini-2.5-flash` (fast) or `gemini-1.5-pro` (higher quality).
- Set `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) in your environment for Gemini.
- To accumulate generations across runs, add `--append`. For flaky latency, optionally use `--timeout 20`.
- Alternatives: use `--engine openai:gpt-4o-mini` with `OPENAI_API_KEY` or `--engine anthropic:claude-3-5-sonnet` with `ANTHROPIC_API_KEY`.
- Output contains either prompts only or prompt/response pairs (when `--generate-refusals`).

## 2) Prepare SFT Dataset (Benign + Adversarial-Refusal)

Inputs:
- Benign helpful pairs (e.g., Alpaca/Dolly/OpenAssistant) in JSONL with either:
  - `{"prompt": "...", "response": "..."}` or
  - `{"instruction": "...", "input": "...", "output": "..."}`
- Adversarial prompts with safe refusals (from step 1) as `{"prompt": "...", "response": "Sorry, I can't..."}`.

Create a balanced mix and splits:

```bash
python src/data/prepare_sft_datasets.py \
  --benign-file path/to/benign.jsonl \
  --adversarial-file data/processed/synthetic/adversarial.jsonl \
  --output data/processed/sft \
  --adversarial-ratio 0.5 \
  --seed 42
```

Outputs:
- `data/processed/sft/train.jsonl`, `val.jsonl`, `test.jsonl` with unified schema:
  - `{"prompt": "...", "response": "...", "source": "benign|adversarial"}`

Tips:
- Try `--adversarial-ratio` in {0.25, 0.5, 0.7} to tune helpfulness vs robustness.

## 3) Train SFT with Tinker (LoRA/QLoRA)

Example (start small, e.g., Llama-3.2-1B):

```bash
python src/training/train_sft_tinker.py \
  --model meta-llama/Llama-3.2-1B \
  --train-file data/processed/sft/train.jsonl \
  --val-file data/processed/sft/val.jsonl \
  --epochs 3 \
  --batch-size 8 \
  --learning-rate 5e-5 \
  --save-steps 200 \
  --lora-rank 32
```

Environment:
- `TINKER_API_KEY` required. Optional `HF_TOKEN` for gated datasets.

## 4) Evaluate Baseline vs SFT

Baseline (zero-shot base model):
```bash
python src/evaluation/evaluate_tinker.py \
  --model-path meta-llama/Llama-3.2-1B \
  --test-file data/processed/advbench/test.jsonl \
  --output-dir results/baseline
```

After SFT:
```bash
python src/evaluation/evaluate_tinker.py \
  --model-path tinker://Llama-3.2-1B-sft-final \
  --test-file data/processed/advbench/test.jsonl \
  --output-dir results/sft
```

Track:
- Refusal rate (higher is safer), ASR (lower is better), benign helpfulness on a held-out benign set if available.

## 5) Data-Mix Experiments
- Re-run step 2 with different `--adversarial-ratio` values.
- Re-train and compare:
  - Benign helpful accuracy (or BLEU/ROUGE on benign test)
  - Jailbreak refusal rate on adversarial tests

Recommended grid:
- Models: `meta-llama/Llama-3.2-1B`, `meta-llama/Llama-3.2-3B`, `Qwen/Qwen3-7B-Instruct` (if available)
- Ratios: `0.25, 0.5, 0.7`
- Epochs: `2-3` with early stopping using val SFT loss.


