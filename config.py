"""
Configuration file for the prompt injection defense project.
Centralized settings for models, training, and evaluation.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
JAILBREAK_DIR = DATA_DIR / "jailbreak"
BENIGN_DIR = DATA_DIR / "benign"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

# Data files
JAILBREAK_FILE = JAILBREAK_DIR / "jailbreak_samples.json"
BENIGN_FILE = BENIGN_DIR / "benign_samples.json"
SFT_TRAINING_DATA = DATA_DIR / "sft_training_data.json"
PREFERENCE_DATA = DATA_DIR / "preference_data.json"

# Model settings
BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"  # Ungated, good default
MODEL_VARIANTS = {
    # Ungated models (no HuggingFace auth required)
    "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "phi-3": "microsoft/Phi-3-mini-4k-instruct",
    
    # Gated models (require HuggingFace authentication)
    "llama3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",  # Gated!
}

# Training hyperparameters
TRAINING_CONFIG = {
    "sft": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "learning_rate": 2e-4,
        "num_epochs": 3,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "max_length": 512,
        "warmup_ratio": 0.1,
    },
    "dpo": {
        "lora_rank": 32,
        "lora_alpha": 64,
        "learning_rate": 1e-4,
        "num_epochs": 2,
        "batch_size": 2,
        "beta": 0.1,  # DPO temperature parameter
        "max_length": 512,
    }
}

# Evaluation settings
EVAL_CONFIG = {
    "num_jailbreak_samples": 100,
    "num_benign_samples": 100,
    "max_tokens": 256,
    "temperature": 0.0,  # Use 0 for deterministic, reproducible benchmarking
}

# Dataset sizes
DATASET_SIZES = {
    "jailbreak_samples": 500,
    "benign_samples": 1000,
}

# Success criteria (from proposal)
SUCCESS_CRITERIA = {
    "min_asr_reduction": 0.30,  # 30% relative reduction
    "max_utility_loss": 0.10,   # 10% max utility loss
}

# Tinker settings
USE_TINKER = bool(os.getenv("TINKER_API_KEY"))
TINKER_API_KEY = os.getenv("TINKER_API_KEY")

# Create necessary directories
for directory in [DATA_DIR, JAILBREAK_DIR, BENIGN_DIR, 
                  CHECKPOINT_DIR, RESULTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

