DSC291 - Prompt Injection Defense Project
==========================================

Post-Training Alignment Against Prompt Injection
Using Llama-3-8B with SFT and RL methods

Project Structure:
------------------
DSC291/
├── requirements.txt           # Python dependencies
├── setup.sh                   # Environment setup script
├── download_data.py           # Download JailbreakBench & benign data
├── prepare_training_data.py   # Format data for SFT/RL training
├── train_sft_tinker.py        # SFT training with Tinker/LoRA
├── evaluate.py                # Evaluation script (ASR, helpfulness)
├── run_pipeline.py            # End-to-end pipeline runner
├── quick_test.py              # Quick test on sample prompts
├── data/                      # Dataset directory
│   ├── jailbreak/            # Adversarial prompts
│   └── benign/               # Normal instructions
├── checkpoints/               # Model checkpoints
└── results/                   # Evaluation results

Quick Start:
------------

1. Install dependencies:
   pip install -r requirements.txt

2. (Optional) Set Tinker API key if using Tinker:
   export TINKER_API_KEY=your_api_key_here

3. Download and prepare data:
   python download_data.py
   python prepare_training_data.py

4. Train SFT baseline:
   python train_sft_tinker.py --num_epochs 3

5. Evaluate the model:
   python evaluate.py --model_path checkpoints/sft_lora

6. Quick test:
   python quick_test.py

Or run the complete pipeline:
   python run_pipeline.py

Notes:
------
- The implementation includes fallback to local training if Tinker API is not available
- Uses LoRA for efficient fine-tuning
- Evaluates Attack Success Rate (ASR) and over-refusal on benign tasks
- Default model: meta-llama/Meta-Llama-3-8B-Instruct

For questions or issues, refer to the project proposal document.

