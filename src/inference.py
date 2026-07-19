# src/inference.py
# Autoregressive Text Generation with GQE Transformer

import torch
import torch.nn.functional as F
import yaml
import argparse
from transformers import AutoTokenizer
from typing import Dict, Any

from src.model import GQETransformerLM

@torch.no_grad()
def generate_text(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
    top_k: int = 50,
    device: torch.device = torch.device("cpu")
) -> str:
    """Generates text autoregressively from a prompt using the model and KV caching.
    
    Args:
        model: GQETransformerLM instance
        tokenizer: HuggingFace Tokenizer instance
        prompt: Input text string
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0 for greedy decoding)
        top_k: Top-k filtering threshold (0 to disable)
        device: Torch device (cpu or cuda)
        
    Returns:
        Generated text string.
    """
    model.eval()
    
    # Tokenize input prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    # Process the prompt (Prefill phase) to populate the initial KV cache
    # logits shape: (1, prompt_len, vocab_size)
    # past_key_values: list of (key_cache, value_cache) per layer, where cache shape is (1, prompt_len, G, d_head)
    logits, past_key_values, _ = model(input_ids, use_cache=True)
    
    # Extract logits for the last token in the prompt
    next_token_logits = logits[:, -1, :]
    
    generated_ids = []
    
    for _ in range(max_new_tokens):
        # Apply temperature and top-k filtering if sampling
        if temperature > 0.0:
            scaled_logits = next_token_logits / temperature
            if top_k > 0:
                # Keep only top_k tokens
                v, _ = torch.topk(scaled_logits, min(top_k, scaled_logits.size(-1)))
                scaled_logits[scaled_logits < v[:, [-1]]] = -float('Inf')
            probs = torch.softmax(scaled_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
        generated_ids.append(next_token.item())
        
        # Stop generating if we hit the End-Of-Sequence (EOS) token
        if next_token.item() == tokenizer.eos_token_id:
            break
            
        # Decode the next step using only the single new token and the accumulated past_key_values
        # This leverages the cached history instead of reprocessing the whole sequence.
        logits, past_key_values, _ = model(next_token, past_key_values=past_key_values, use_cache=True)
        next_token_logits = logits[:, -1, :]
        
    # Decode full sequence
    return prompt + tokenizer.decode(generated_ids)


def run_generation(config_path: str, checkpoint_path: str, prompt: str, max_tokens: int, temp: float, top_k: int):
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on device: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["data"].get("tokenizer", "gpt2"))
    config["model"]["vocab_size"] = tokenizer.vocab_size
    
    # Initialize model
    model = GQETransformerLM(config)
    
    if checkpoint_path:
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint provided. Running generation with randomly initialized weights (for testing structure).")
        
    model.to(device)
    
    print(f"\nPrompt: '{prompt}'")
    print(f"Generating (max {max_tokens} new tokens, temp={temp}, top_k={top_k})...")
    
    generated_text = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_tokens,
        temperature=temp,
        top_k=top_k,
        device=device
    )
    
    print("\n--- Generated Text Output ---")
    print(generated_text)
    print("-----------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GQE Interactive Generation / Inference")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint .pt (optional, falls back to random init)")
    parser.add_argument("--prompt", type=str, default="The future of artificial intelligence is", help="Prompt text")
    parser.add_argument("--max_tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--temp", type=float, default=0.7, help="Sampling temperature (0.0 for greedy)")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering threshold")
    args = parser.parse_args()
    
    run_generation(args.config, args.checkpoint, args.prompt, args.max_tokens, args.temp, args.top_k)
