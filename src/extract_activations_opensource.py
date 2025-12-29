#!/usr/bin/env python3
"""
Activation extraction script for open-source models.
Extracts activations from transformer layers for empathy probe training.
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import os

# Model configurations
MODEL_CONFIGS = {
    'llama-70b': 'meta-llama/Llama-3.1-70B-Instruct',
    'gemma-27b': 'google/gemma-2-27b-it',
    'qwen-32b': 'Qwen/Qwen2.5-32B-Instruct',
    'yi-34b': '01-ai/Yi-1.5-34B-Chat',
    'mistral-24b': 'mistralai/Mistral-Small-3.1-24B-Instruct-2503'
}


def extract_activations(model, tokenizer, texts: List[str], layer_idx: int, 
                       pooling_method: str = "last_token") -> torch.Tensor:
    """
    Extract activations from a specific layer for a list of texts.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        texts: List of input texts
        layer_idx: Layer index to extract from
        pooling_method: "last_token" or "mean_pool"
        
    Returns:
        torch.Tensor: Activations of shape [num_texts, hidden_dim]
    """
    activations = []

    def hook(module, input, output):
        hidden_states = output[0]  # [batch, seq_len, hidden_dim]
        
        if pooling_method == "last_token":
            # Use last token (final position) activation
            act = hidden_states[:, -1, :]  # [batch, hidden_dim]
        elif pooling_method == "mean_pool":
            # Mean pool over sequence length
            act = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        else:
            raise ValueError(f"Unknown pooling method: {pooling_method}")
            
        activations.append(act.detach().cpu())

    # Register hook on the specified layer
    handle = model.model.layers[layer_idx].register_forward_hook(hook)

    with torch.no_grad():
        for text in tqdm(texts, desc=f"Processing layer {layer_idx}"):
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            model(**inputs)

    handle.remove()
    return torch.cat(activations, dim=0)


def load_contrastive_pairs(model_name: str) -> Tuple[List[str], List[int]]:
    """
    Load contrastive pairs for a specific model.
    
    Args:
        model_name: Name of the model to load pairs for
        
    Returns:
        Tuple of (texts, labels) where labels are 1 for empathic, 0 for non-empathic
    """
    data_path = Path(f"data/contrastive_pairs/{model_name}_contrastive_pairs.jsonl")
    
    if not data_path.exists():
        raise FileNotFoundError(f"Contrastive pairs not found for {model_name} at {data_path}")
    
    texts = []
    labels = []
    
    with open(data_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Add empathic response
            texts.append(data['empathic_response'])
            labels.append(1)
            # Add non-empathic response
            texts.append(data['non_empathic_response'])
            labels.append(0)
    
    return texts, labels


def extract_all_layers(model, tokenizer, texts: List[str], layer_indices: List[int], 
                      save_path: Path, pooling_method: str = "last_token", batch_size: int = 1) -> None:
    """
    Extract activations from multiple layers and save to disk.
    
    Args:
        model: The transformer model
        tokenizer: The tokenizer
        texts: List of input texts
        layer_indices: List of layer indices to extract from
        save_path: Path to save activations
        pooling_method: "last_token" or "mean_pool"
        batch_size: Batch size for processing (keep at 1 for large models)
    """
    save_path.mkdir(parents=True, exist_ok=True)
    
    for layer_idx in layer_indices:
        print(f"Extracting activations from layer {layer_idx}...")
        
        # Extract activations for this layer
        activations = extract_activations(model, tokenizer, texts, layer_idx, pooling_method)
        
        # Save activations
        output_file = save_path / f"layer_{layer_idx}.pt"
        torch.save(activations, output_file)
        print(f"Saved layer {layer_idx} activations to {output_file}")
        
        # Clear memory
        del activations
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Extract activations from transformer layers")
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()),
                       help="Model to extract activations from")
    parser.add_argument("--layers", type=str, default=None,
                       help="Comma-separated layer indices (e.g., '10,20,30') or 'all' for all layers")
    parser.add_argument("--all-layers", action="store_true",
                       help="Extract from all layers")
    parser.add_argument("--output-dir", type=str, default="data/activations",
                       help="Output directory for activations")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--bf16", action="store_true", default=True,
                       help="Use BF16 precision")
    parser.add_argument("--pooling", type=str, default="last_token", choices=["last_token", "mean_pool"],
                       help="Pooling method for sequence activations")
    
    args = parser.parse_args()
    
    # Set up device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model_id = MODEL_CONFIGS[args.model]
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate precision
    torch_dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True
    )
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Load contrastive pairs
    print(f"Loading contrastive pairs for {args.model}...")
    texts, labels = load_contrastive_pairs(args.model)
    print(f"Loaded {len(texts)} texts ({len(texts)//2} pairs)")
    
    # Determine which layers to extract
    num_layers = model.config.num_hidden_layers
    
    if args.all_layers:
        layer_indices = list(range(num_layers))
    elif args.layers:
        if args.layers.lower() == "all":
            layer_indices = list(range(num_layers))
        else:
            layer_indices = [int(x.strip()) for x in args.layers.split(",")]
    else:
        # Default: extract from a few key layers
        layer_indices = [
            num_layers // 4,      # Early layer
            num_layers // 2,      # Middle layer  
            3 * num_layers // 4,  # Late layer
            num_layers - 1        # Final layer
        ]
    
    print(f"Extracting from layers: {layer_indices}")
    
    # Set up output path
    output_path = Path(args.output_dir) / args.model
    
    # Extract activations
    extract_all_layers(model, tokenizer, texts, layer_indices, output_path, args.pooling)
    
    # Save labels
    labels_file = output_path / "labels.pt"
    torch.save(torch.tensor(labels), labels_file)
    print(f"Saved labels to {labels_file}")
    
    # Save metadata
    metadata = {
        "model": args.model,
        "model_id": model_id,
        "num_texts": len(texts),
        "num_pairs": len(texts) // 2,
        "layers": layer_indices,
        "num_layers": num_layers,
        "pooling_method": args.pooling,
        "torch_dtype": str(torch_dtype),
        "device": device
    }
    
    metadata_file = output_path / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_file}")
    
    print("Activation extraction complete!")


if __name__ == "__main__":
    main()