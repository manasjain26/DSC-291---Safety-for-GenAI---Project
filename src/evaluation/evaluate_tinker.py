"""
Evaluate trained models using Tinker sampling client.
"""

import argparse
import json
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.tinker_api import TinkerTrainingClient, SafetyEvaluator
from dotenv import load_dotenv

load_dotenv()


def load_test_prompts(test_file: str):
    """Load test prompts from JSONL file."""
    prompts = []
    with open(test_file, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            prompts.append(data['prompt'])
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Evaluate model with Tinker")
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Model path: base model (e.g., meta-llama/Llama-3.2-1B) or saved checkpoint (e.g., Llama-3.2-1B-dpo-final)"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        required=True,
        help="Path to test JSONL file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Maximum tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_file}...")
    test_prompts = load_test_prompts(args.test_file)
    print(f"Loaded {len(test_prompts)} test prompts")
    
    # Initialize sampling client
    print(f"\nInitializing Tinker client for {args.model_path}...")
    
    # Import here to avoid circular dependency
    import tinker
    from transformers import AutoTokenizer
    
    service_client = tinker.ServiceClient()
    
    # Check if it's a saved checkpoint or base model
    if args.model_path.startswith("tinker://"):
        # Direct tinker:// path
        model_path = args.model_path
        print(f"Loading saved model from {model_path}...")
        base_model_for_tokenizer = "meta-llama/Llama-3.2-1B"  # Default
    elif "/" in args.model_path and not args.model_path.startswith("meta-") and not args.model_path.startswith("google/"):
        # Looks like a saved checkpoint name (e.g., "Llama-3.2-1B-dpo-final")
        # Get the training client ID from the saved state
        print(f"Loading from saved checkpoint: {args.model_path}")
        # We need to create a temporary training client to get the path
        # For now, construct the path manually
        # The actual path would be like tinker://<training_client_id>/<checkpoint_name>
        print("Note: Please provide the full tinker:// path including training client ID")
        print(f"Example: tinker://<your-training-client-id>/{args.model_path}")
        return
    else:
        # Base model - create training client and save immediately for sampling
        print(f"Creating new training client for base model: {args.model_path}")
        training_client = service_client.create_lora_training_client(
            base_model=args.model_path,
            rank=32
        )
        result = training_client.save_weights_for_sampler(name="eval-temp").result()
        model_path = result.path
        base_model_for_tokenizer = args.model_path
        print(f"Saved to {model_path} for evaluation")
    
    # Create sampling client
    sampling_client = service_client.create_sampling_client(model_path=model_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_for_tokenizer)
    
    # Wrap in our client
    from src.utils.tinker_api import TinkerSamplingClient
    wrapped_client = TinkerSamplingClient(sampling_client, tokenizer)
    
    # Run evaluation
    print("\nRunning safety evaluation...")
    evaluator = SafetyEvaluator(
        sampling_client=wrapped_client,
        output_dir=args.output_dir
    )
    
    results = evaluator.evaluate_jailbreak(
        test_prompts=test_prompts,
        model_name=args.model_path.replace("/", "_").replace(":", "_"),
        max_tokens=args.max_tokens
    )
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {results['model']}")
    print(f"Total prompts: {results['total_prompts']}")
    print(f"Refusal rate: {results['refusal_rate']:.2%}")
    print(f"Attack success rate: {results['attack_success_rate']:.2%}")
    print("="*50)


if __name__ == "__main__":
    main()
