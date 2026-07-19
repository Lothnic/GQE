# src/evaluate.py
# §4.1 — Zero-shot Evaluation on HellaSwag, PIQA, and ARC-Easy

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import argparse
import yaml
from typing import List, Dict, Any

from src.model import GQETransformerLM

@torch.no_grad()
def compute_conditional_log_likelihood(
    model: torch.nn.Module,
    tokenizer: Any,
    context: str,
    continuation: str,
    device: torch.device
) -> float:
    """Computes the log-likelihood of the continuation conditioned on the context.
    
    P(continuation | context) = prod_{t} P(token_t | tokens_{<t})
    """
    # Tokenize context and full (context + continuation)
    context_ids = tokenizer.encode(context, add_special_tokens=False)
    full_ids = tokenizer.encode(context + continuation, add_special_tokens=False)
    
    # If tokenization results in the same length (no continuation tokens), return very low probability
    if len(full_ids) <= len(context_ids):
        return -float('inf')
        
    # Convert to tensors
    input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
    
    # Forward pass through model
    logits, _, _ = model(input_ids)
    
    # Calculate log probabilities
    # Shape: (1, seq_len, vocab_size)
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Shift logits and targets to align predictions with actual next tokens
    # We only care about prediction of the continuation tokens
    # continuation tokens start at index len(context_ids) in full_ids
    # they are predicted at logits index len(context_ids) - 1 to len(full_ids) - 2
    cont_length = len(full_ids) - len(context_ids)
    
    # Gather target log probabilities
    target_ids = input_ids[0, len(context_ids):] # Shape: (cont_length,)
    pred_log_probs = log_probs[0, len(context_ids)-1 : len(full_ids)-1, :] # Shape: (cont_length, vocab_size)
    
    # Extract the log probability of the actual target token at each position
    target_log_probs = pred_log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
    
    # Return sum of log-likelihoods
    return target_log_probs.sum().item()


def evaluate_piqa(model: torch.nn.Module, tokenizer: Any, device: torch.device, max_samples: int = 100) -> float:
    """Evaluates the model on PIQA (Physical Commonsense Reasoning).
    
    PIQA fields: goal, sol1, sol2, label (0 or 1)
    """
    print(f"Loading PIQA validation split...")
    dataset = load_dataset("ybisk/piqa", split="validation", trust_remote_code=True)
    
    correct = 0
    total = 0
    
    # Limit samples for speed if requested
    samples = list(dataset)[:max_samples] if max_samples > 0 else list(dataset)
    
    for sample in tqdm(samples, desc="Evaluating PIQA"):
        goal = sample["goal"]
        sol1 = sample["sol1"]
        sol2 = sample["sol2"]
        label = sample["label"] # 0 or 1
        
        # Calculate log-likelihood for each solution option
        ll1 = compute_conditional_log_likelihood(model, tokenizer, goal + " ", sol1, device)
        ll2 = compute_conditional_log_likelihood(model, tokenizer, goal + " ", sol2, device)
        
        prediction = 0 if ll1 > ll2 else 1
        if prediction == label:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0.0
    print(f"PIQA Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy


def evaluate_arc_easy(model: torch.nn.Module, tokenizer: Any, device: torch.device, max_samples: int = 100) -> float:
    """Evaluates the model on ARC-Easy (Science QA).
    
    ARC fields: question, choices: {'text': List[str], 'label': List[str]}, answerKey
    """
    print(f"Loading ARC-Easy validation split...")
    dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation", trust_remote_code=True)
    
    correct = 0
    total = 0
    
    # Limit samples
    samples = list(dataset)[:max_samples] if max_samples > 0 else list(dataset)
    
    for sample in tqdm(samples, desc="Evaluating ARC-Easy"):
        question = sample["question"]
        choices = sample["choices"]
        answer_key = sample["answerKey"]
        
        labels = choices["label"]
        texts = choices["text"]
        
        # Calculate log likelihood for each choices
        lls = []
        for choice_text in texts:
            ll = compute_conditional_log_likelihood(model, tokenizer, question + " ", choice_text, device)
            lls.append(ll)
            
        # Predict the label with max log-likelihood
        max_idx = lls.index(max(lls))
        pred_label = labels[max_idx]
        
        if pred_label == answer_key:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0.0
    print(f"ARC-Easy Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy


def evaluate_hellaswag(model: torch.nn.Module, tokenizer: Any, device: torch.device, max_samples: int = 100) -> float:
    """Evaluates the model on HellaSwag (Commonsense NLG).
    
    HellaSwag fields: ctx, endings (list of 4 strings), label (str representation of index)
    """
    print(f"Loading HellaSwag validation split...")
    dataset = load_dataset("Rowan/hellaswag", split="validation", trust_remote_code=True)
    
    correct = 0
    total = 0
    
    # Limit samples
    samples = list(dataset)[:max_samples] if max_samples > 0 else list(dataset)
    
    for sample in tqdm(samples, desc="Evaluating HellaSwag"):
        ctx = sample["ctx"]
        endings = sample["endings"] # List of 4 options
        label = int(sample["label"]) # correct index
        
        lls = []
        for ending in endings:
            ll = compute_conditional_log_likelihood(model, tokenizer, ctx + " ", ending, device)
            lls.append(ll)
            
        prediction = lls.index(max(lls))
        if prediction == label:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0.0
    print(f"HellaSwag Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy


def run_evaluation(config_path: str, checkpoint_path: str, max_samples: int = 100):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["data"].get("tokenizer", "gpt2"))
    
    # Load model configuration & load weights
    config["model"]["vocab_size"] = tokenizer.vocab_size
    model = GQETransformerLM(config)
    
    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    
    # Run tasks
    piqa_acc = evaluate_piqa(model, tokenizer, device, max_samples)
    arc_acc = evaluate_arc_easy(model, tokenizer, device, max_samples)
    hella_acc = evaluate_hellaswag(model, tokenizer, device, max_samples)
    
    print(f"\n--- Evaluation Results Summary ---")
    print(f"PIQA: {piqa_acc * 100:.2f}%")
    print(f"ARC-Easy: {arc_acc * 100:.2f}%")
    print(f"HellaSwag: {hella_acc * 100:.2f}%")
    print(f"Average: {((piqa_acc + arc_acc + hella_acc) / 3) * 100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot Evaluation for GQE")
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint .pt")
    parser.add_argument("--max_samples", type=int, default=100, help="Max validation samples per dataset (0 for all)")
    args = parser.parse_args()
    
    run_evaluation(args.config, args.checkpoint, args.max_samples)
