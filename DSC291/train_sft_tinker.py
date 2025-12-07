"""
Supervised Fine-Tuning (SFT) using Tinker for Llama-3-8B.
Uses LoRA for parameter-efficient fine-tuning on adversarial + benign data.
"""

import os
import json
import argparse
from typing import List, Dict
import tinker
from tinker import TensorData
import numpy as np


def load_training_data(data_file: str) -> List[Dict]:
    """Load training data from JSON file."""
    with open(data_file, 'r') as f:
        data = json.load(f)
    return data


def format_prompt(instruction: str, model_name: str = "llama3") -> str:
    """
    Format prompt according to model's chat template.
    Llama-3 uses a specific format with special tokens.
    """
    if "llama" in model_name.lower():
        # Llama-3 chat format
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    else:
        # Generic format
        return f"### Instruction:\n{instruction}\n\n### Response:\n"


def prepare_training_samples(data: List[Dict], model_name: str = "llama3"):
    """
    Prepare training samples in the format expected by Tinker.
    Returns list of (input, target) tuples.
    """
    samples = []
    for item in data:
        instruction = item["instruction"]
        response = item["response"]
        
        # Format as chat-style prompt
        prompt = format_prompt(instruction, model_name)
        full_text = prompt + response + "<|eot_id|>"
        
        samples.append({
            "text": full_text,
            "type": item.get("type", "unknown")
        })
    
    return samples


def train_sft_with_tinker(
    data_file: str = "data/sft_training_data.json",
    base_model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
    output_dir: str = "checkpoints/sft_lora",
    lora_rank: int = 32,
    learning_rate: float = 2e-4,
    num_epochs: int = 3,
    batch_size: int = 4,
):
    """
    Train SFT model using Tinker API.
    
    IMPORTANT: This function trains on the training split and evaluates on 
    the held-out test split to prevent data leakage. The test sets are:
    - data/jailbreak/jailbreak_test.json
    - data/benign/benign_test.json
    """
    print("=" * 60)
    print("Starting SFT Training with Tinker")
    print("=" * 60)
    print(f"Base model: {base_model}")
    print(f"Training data: {data_file}")
    print(f"LoRA rank: {lora_rank}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    print("=" * 60)
    
    # Check for API key
    if not os.getenv("TINKER_API_KEY"):
        print("\n❌ ERROR: TINKER_API_KEY not found!")
        print("Please set your API key: setenv TINKER_API_KEY your_key_here")
        return None
    
    try:
        # Initialize Tinker client
        print("\nInitializing Tinker service client...")
        service_client = tinker.ServiceClient()
        
        # Create LoRA training client
        print(f"Creating LoRA training client for {base_model}...")
        training_client = service_client.create_lora_training_client(
            base_model=base_model,
            rank=lora_rank
        )
        
        # Load tokenizer (needed to encode text to ModelInput)
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)
        
        # Load and prepare data
        print("\nLoading training data...")
        raw_data = load_training_data(data_file)
        training_samples = prepare_training_samples(raw_data, "llama3")
        
        print(f"Loaded {len(training_samples)} training samples")
        adversarial_count = sum(1 for s in training_samples if s["type"] == "adversarial")
        benign_count = sum(1 for s in training_samples if s["type"] == "benign")
        print(f"  - Adversarial: {adversarial_count}")
        print(f"  - Benign: {benign_count}")
        
        # Training loop
        print("\nStarting training...")
        for epoch in range(num_epochs):
            print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
            
            total_loss = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(training_samples), batch_size):
                batch = training_samples[i:i + batch_size]
                batch_texts = [s["text"] for s in batch]
                
                # Forward and backward pass
                try:
                    # Convert texts to Datum objects (required by Tinker)
                    # Following format from: https://github.com/manasjain26/DSC-291---Safety-for-GenAI---Project/
                    batch_data = []
                    for text in batch_texts:
                        # Encode text to token IDs
                        encoded = tokenizer.encode(text, add_special_tokens=True)
                        
                        # For causal LM: input is tokens[:-1], target is tokens[1:]
                        input_tokens = encoded[:-1]
                        target_tokens = encoded[1:]
                        
                        # Create weights (1.0 for all tokens to train on full sequence)
                        weights = np.ones(len(target_tokens), dtype=np.float32)
                        
                        # Create Datum with proper loss_fn_inputs format
                        datum = tinker.types.Datum(
                            model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
                            loss_fn_inputs=dict(
                                target_tokens=TensorData.from_numpy(np.array(target_tokens, dtype=np.int32)),
                                weights=TensorData.from_numpy(weights)
                            )
                        )
                        batch_data.append(datum)
                    
                    # Use "cross_entropy" as the loss function (built-in for causal LM)
                    future = training_client.forward_backward(batch_data, "cross_entropy")
                    result = future.result()  # Wait for completion

                    
                    # Extract loss from result
                    # loss_fn_outputs contains dict with 'elementwise_loss' (TensorData with per-token losses)
                    if hasattr(result, 'loss_fn_outputs') and result.loss_fn_outputs:
                        batch_losses = []
                        for output in result.loss_fn_outputs:
                            # Get elementwise_loss TensorData
                            if 'elementwise_loss' in output:
                                elementwise_loss = output['elementwise_loss'].data  # numpy array
                                # Average across tokens (each sequence has different length)
                                sequence_loss = float(np.mean(elementwise_loss))
                                batch_losses.append(sequence_loss)
                        
                        # Average loss across batch
                        loss_value = sum(batch_losses) / len(batch_losses) if batch_losses else 0.0
                    else:
                        loss_value = 0.0
                    
                    # Perform optimizer step with Adam parameters
                    adam_params = tinker.types.AdamParams(learning_rate=learning_rate)
                    training_client.optim_step(adam_params).result()
                    
                    total_loss += loss_value
                    num_batches += 1
                    
                    if num_batches % 10 == 0 and num_batches > 0:
                        avg_loss = total_loss / num_batches
                        print(f"  Batch {num_batches}: avg_loss = {avg_loss:.4f}")
                        
                except Exception as e:
                    print(f"  Error in batch {num_batches}: {e}")
                    continue
            
            avg_epoch_loss = total_loss / max(num_batches, 1)
            print(f"Epoch {epoch + 1} complete. Average loss: {avg_epoch_loss:.4f}")
        
        # Save weights and get sampling client
        print(f"\nSaving model...")
        model_name = f"sft_model_{base_model.replace('/', '_')}"
        sampling_client = training_client.save_weights_and_get_sampling_client(name=model_name)
        
        print("\n" + "=" * 60)
        print("SFT Training Complete!")
        print("=" * 60)
        
        # Evaluate immediately in the same session with LLM judge
        print("\nStarting evaluation with LLM judge...")
        from evaluate import evaluate_jailbreak_robustness, evaluate_benign_helpfulness, load_eval_data
        from llm_judge import LLMJudge
        
        # Initialize LLM judge
        judge_model = "Qwen/Qwen3-235B-A22B-Instruct-2507"
        print(f"🔍 Initializing LLM Judge: {judge_model}")
        llm_judge = LLMJudge(judge_model=judge_model)
        
        # Load TEST data (held-out, not seen during training)
        print("\n⚠️  Using held-out TEST data for evaluation (not training data)")
        jailbreak_data, benign_data = load_eval_data(
            "data/jailbreak/jailbreak_test.json",
            "data/benign/benign_test.json"
        )
        
        # Evaluate jailbreak robustness with LLM judge
        jailbreak_results = evaluate_jailbreak_robustness(
            sampling_client, jailbreak_data, use_tinker=True, 
            model=None, tokenizer=tokenizer, llm_judge=llm_judge
        )
        
        # Evaluate benign helpfulness
        benign_results = evaluate_benign_helpfulness(
            sampling_client, benign_data, use_tinker=True,
            model=None, tokenizer=tokenizer
        )
        
        # Save results (includes all responses)
        os.makedirs("results", exist_ok=True)
        results = {
            "model_name": model_name,
            "jailbreak_robustness": jailbreak_results,
            "benign_helpfulness": benign_results
        }
        
        with open("results/sft_evaluation.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n" + "=" * 60)
        print("EVALUATION COMPLETE!")
        print(f"📊 ASR: {jailbreak_results['asr']:.2f}%")
        print(f"📊 Over-refusal: {benign_results['over_refusal_rate']:.2f}%")
        print("Results saved to: results/sft_evaluation.json")
        print("=" * 60)
        
        return model_name
        
    except Exception as e:
        print(f"\n❌ Error during Tinker training: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT training with Tinker")
    parser.add_argument("--data_file", type=str, default="data/sft_training_data.json",
                        help="Path to training data JSON file")
    parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.2-1B",
                        help="Base model to fine-tune")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_lora",
                        help="Output directory for checkpoints")
    parser.add_argument("--lora_rank", type=int, default=32,
                        help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size")
    
    args = parser.parse_args()
    
    train_sft_with_tinker(
        data_file=args.data_file,
        base_model=args.base_model,
        output_dir=args.output_dir,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size
    )

