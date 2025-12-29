#!/usr/bin/env python3
"""
Cross-Model Probe Transfer Analysis (Issue #5)

Tests empathy probe transferability across 5 different AI model architectures.
Generates a 5x5 transfer matrix and heatmap visualization.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
PROBES_DIR = PROJECT_ROOT / "results" / "probes"
RESULTS_DIR = PROJECT_ROOT / "results" / "transfer_analysis"
FIGURES_DIR = PROJECT_ROOT / "figures"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations matching probe_extraction_cross_model.py
MODELS = {
    "llama-70b": {
        "name": "meta-llama/Llama-3.1-70B-Instruct",
        "architecture": "LLaMA",
        "layers": [10, 20, 30, 40, 50, 60, 70],
        "hidden_size": 8192
    },
    "gemma-27b": {
        "name": "google/gemma-2-27b-it",
        "architecture": "Gemma", 
        "layers": [6, 12, 18, 24, 30, 36, 42],
        "hidden_size": 4608
    },
    "qwen-32b": {
        "name": "Qwen/Qwen2.5-32B-Instruct",
        "architecture": "Qwen",
        "layers": [8, 16, 24, 32, 40, 48, 56],
        "hidden_size": 5120
    },
    "yi-34b": {
        "name": "01-ai/Yi-1.5-34B-Chat",
        "architecture": "LLaMA-like",
        "layers": [8, 16, 24, 32, 40, 48, 56],
        "hidden_size": 7168
    },
    "mistral-24b": {
        "name": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        "architecture": "Mistral",
        "layers": [6, 12, 18, 24, 30, 36],
        "hidden_size": 8192
    }
}


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer with BF16 precision."""
    logger.info(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # TODO: see padding strategy
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()
    
    logger.info(f"✓ Loaded {model_name}")
    return model, tokenizer


def load_probe_direction(source_model: str, layer: int) -> Optional[np.ndarray]:
    """Load probe direction from saved file."""
    probe_path = PROBES_DIR / f"{source_model}_layer{layer}_probe.npy"
    
    if not probe_path.exists():
        logger.warning(f"Probe not found: {probe_path}")
        return None
    
    return np.load(probe_path)


def get_activations(model, tokenizer, texts: List[str], layer: int, device: str = "cuda") -> np.ndarray:
    """Extract activations from specified layer for list of texts."""
    activations = []
    
    for text in tqdm(texts, desc=f"Layer {layer}", leave=False):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        # Get hidden states from target layer
        hidden_states = outputs.hidden_states[layer]  # (batch, seq_len, hidden_size)
        
        # Mean pool across sequence length
        # TODO: check if we will use mean pooling or sth else
        pooled = hidden_states.mean(dim=1).squeeze()  # (hidden_size,)
        activations.append(pooled.cpu().numpy())
    
    return np.array(activations)


def project_activations(activations: np.ndarray, probe_direction: np.ndarray) -> np.ndarray:
    """Project activations onto probe direction."""
    # Handle dimensionality mismatch by truncating or padding
    if activations.shape[1] != probe_direction.shape[0]:
        min_dim = min(activations.shape[1], probe_direction.shape[0])
        activations = activations[:, :min_dim]
        probe_direction = probe_direction[:min_dim]
        logger.warning(f"Dimension mismatch handled: using first {min_dim} dimensions")
    
    return activations @ probe_direction


def evaluate_transfer(
    source_model: str,
    target_model: str, 
    source_layer: int,
    target_layer: int,
    device: str = "cuda"
) -> Dict:
    """Evaluate probe transfer between two models at specific layers."""
    logger.info(f"Transfer: {source_model}[{source_layer}] → {target_model}[{target_layer}]")
    
    # Load probe direction from source model
    probe_direction = load_probe_direction(source_model, source_layer)
    if probe_direction is None:
        return {"auroc": 0.0, "error": "Probe not found"}
    
    # Load target model
    target_config = MODELS[target_model]
    target_model_obj, tokenizer = load_model_and_tokenizer(target_config["name"], device)
    
    # Load test dataset
    empathic_texts, non_empathic_texts = load_test_dataset()
    
    try:
        # Extract activations from target model
        emp_acts = get_activations(target_model_obj, tokenizer, empathic_texts, target_layer, device)
        non_acts = get_activations(target_model_obj, tokenizer, non_empathic_texts, target_layer, device)
        
        # Project onto source probe direction
        emp_proj = project_activations(emp_acts, probe_direction)
        non_proj = project_activations(non_acts, probe_direction)
        
        # Compute AUROC
        y_true = np.array([1] * len(emp_proj) + [0] * len(non_proj))
        y_scores = np.concatenate([emp_proj, non_proj])
        
        auroc = roc_auc_score(y_true, y_scores)
        
        result = {
            "auroc": float(auroc),
            "source_model": source_model,
            "target_model": target_model,
            "source_layer": source_layer,
            "target_layer": target_layer,
            "empathic_mean": float(emp_proj.mean()),
            "non_empathic_mean": float(non_proj.mean()),
            "separation": float(abs(emp_proj.mean() - non_proj.mean()))
        }
        
        logger.info(f"AUROC: {auroc:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"Transfer evaluation failed: {e}")
        return {"auroc": 0.0, "error": str(e)}
    
    finally:
        # Clean up GPU memory
        del target_model_obj
        torch.cuda.empty_cache()


def load_test_dataset() -> Tuple[List[str], List[str]]:
    """Load test dataset for transfer evaluation."""
    file_path = DATA_DIR / "merged_cleaned_pairs.jsonl"
    
    empathic_texts = []
    non_empathic_texts = []
    
    with open(file_path, 'r') as f:
        for line in f:
            pair = json.loads(line)
            empathic_texts.append(pair['empathic_text'])
            non_empathic_texts.append(pair['non_empathic_text'])
    
    # Use subset for transfer testing to save time
    subset_size = min(100, len(empathic_texts))
    return empathic_texts[:subset_size], non_empathic_texts[:subset_size]


def find_best_layers(models_to_test: List[str]) -> Dict[str, int]:
    """Find best performing layer for each model from previous results."""
    best_layers = {}
    
    for model in models_to_test:
        results_path = PROJECT_ROOT / "results" / "cross_model_validation" / f"{model}_results.json"
        
        if results_path.exists():
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            # Find layer with highest AUROC
            best_auroc = 0.0
            best_layer = None
            
            for layer, metrics in results["layers"].items():
                if metrics["auroc"] > best_auroc:
                    best_auroc = metrics["auroc"] 
                    best_layer = int(layer)
            
            if best_layer is not None:
                best_layers[model] = best_layer
                logger.info(f"Best layer for {model}: {best_layer} (AUROC: {best_auroc:.3f})")
            else:
                # Fallback to middle layer
                best_layers[model] = MODELS[model]["layers"][len(MODELS[model]["layers"])//2]
                logger.warning(f"No results found for {model}, using middle layer: {best_layers[model]}")
        else:
            # Fallback to middle layer
            best_layers[model] = MODELS[model]["layers"][len(MODELS[model]["layers"])//2]
            logger.warning(f"No results found for {model}, using middle layer: {best_layers[model]}")
    
    return best_layers


def run_transfer_matrix(models_to_test: List[str], device: str = "cuda") -> Dict:
    """Run complete 5x5 transfer analysis."""
    logger.info(f"Running transfer matrix for models: {models_to_test}")
    
    # Find best layers for each model
    best_layers = find_best_layers(models_to_test)
    
    # Initialize transfer matrix
    transfer_results = {}
    transfer_matrix = np.zeros((len(models_to_test), len(models_to_test)))
    
    for i, source_model in enumerate(models_to_test):
        for j, target_model in enumerate(models_to_test):
            source_layer = best_layers[source_model]
            target_layer = best_layers[target_model]
            
            # Evaluate transfer
            result = evaluate_transfer(source_model, target_model, source_layer, target_layer, device)
            
            # Store results
            key = f"{source_model}→{target_model}"
            transfer_results[key] = result
            transfer_matrix[i, j] = result.get("auroc", 0.0)
    
    # Package results
    matrix_results = {
        "models": models_to_test,
        "best_layers": best_layers,
        "transfer_matrix": transfer_matrix.tolist(),
        "detailed_results": transfer_results,
        "architectures": {model: MODELS[model]["architecture"] for model in models_to_test}
    }
    
    return matrix_results


def create_transfer_heatmap(matrix_results: Dict, output_path: Path):
    """Create transfer heatmap visualization."""
    models = matrix_results["models"] 
    transfer_matrix = np.array(matrix_results["transfer_matrix"])
    architectures = matrix_results["architectures"]
    
    # Create labels with architecture info
    labels = [f"{model}\n({architectures[model]})" for model in models]
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    
    # Custom colormap with better contrast
    cmap = sns.color_palette("viridis", as_cmap=True)
    
    # Create heatmap
    ax = sns.heatmap(
        transfer_matrix,
        annot=True,
        fmt='.3f',
        xticklabels=labels,
        yticklabels=labels,
        cmap=cmap,
        vmin=0.0,
        vmax=1.0,
        square=True,
        cbar_kws={'label': 'Transfer AUROC'}
    )
    
    # Styling
    plt.title('Cross-Model Empathy Probe Transfer (5×5 Matrix)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Target Model', fontsize=12, fontweight='bold')
    plt.ylabel('Source Model', fontsize=12, fontweight='bold')
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Highlight diagonal (same-model transfers)
    for i in range(len(models)):
        rect = plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='red', linewidth=3)
        ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Transfer heatmap saved to {output_path}")


def analyze_architecture_similarity(matrix_results: Dict):
    """Analyze transfer performance based on architecture similarity."""
    models = matrix_results["models"]
    transfer_matrix = np.array(matrix_results["transfer_matrix"])
    architectures = matrix_results["architectures"]
    
    logger.info("\n=== Architecture Similarity Analysis ===")
    
    # Test Yi-Llama hypothesis
    yi_idx = models.index("yi-34b") if "yi-34b" in models else None
    llama_idx = models.index("llama-70b") if "llama-70b" in models else None
    
    if yi_idx is not None and llama_idx is not None:
        yi_to_llama = transfer_matrix[yi_idx, llama_idx]
        llama_to_yi = transfer_matrix[llama_idx, yi_idx]
        avg_transfer = (yi_to_llama + llama_to_yi) / 2
        
        logger.info(f"Yi ↔ Llama transfer: {avg_transfer:.3f}")
        logger.info(f"  Yi → Llama: {yi_to_llama:.3f}")
        logger.info(f"  Llama → Yi: {llama_to_yi:.3f}")
    
    # Compute average within vs across architecture transfers
    within_arch_scores = []
    across_arch_scores = []
    
    for i, source_model in enumerate(models):
        for j, target_model in enumerate(models):
            if i != j:  # Skip self-transfers
                score = transfer_matrix[i, j]
                if architectures[source_model] == architectures[target_model]:
                    within_arch_scores.append(score)
                else:
                    across_arch_scores.append(score)
    
    if within_arch_scores:
        logger.info(f"Within-architecture transfer: {np.mean(within_arch_scores):.3f} ± {np.std(within_arch_scores):.3f}")
    if across_arch_scores:
        logger.info(f"Across-architecture transfer: {np.mean(across_arch_scores):.3f} ± {np.std(across_arch_scores):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Cross-model probe transfer analysis")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODELS.keys()) + ["all"],
        default=["all"],
        help="Which models to test (default: all)"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Determine which models to test
    if "all" in args.models:
        models_to_test = list(MODELS.keys())
    else:
        models_to_test = args.models
    
    logger.info(f"Running transfer analysis for: {models_to_test}")
    logger.info(f"Device: {args.device}\n")
    
    # Run transfer matrix analysis
    matrix_results = run_transfer_matrix(models_to_test, args.device)
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "transfer_matrix_results.json"
    with open(results_path, 'w') as f:
        json.dump(matrix_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Create visualization
    heatmap_path = FIGURES_DIR / "cross_model_transfer_heatmap.png"
    create_transfer_heatmap(matrix_results, heatmap_path)
    
    # Analyze architecture effects
    analyze_architecture_similarity(matrix_results)
    
    # Print summary
    transfer_matrix = np.array(matrix_results["transfer_matrix"])
    logger.info(f"\n=== Transfer Matrix Summary ===")
    logger.info(f"Average transfer AUROC: {transfer_matrix.mean():.3f}")
    logger.info(f"Best transfer pair: {transfer_matrix.max():.3f}")
    logger.info(f"Worst transfer pair: {transfer_matrix.min():.3f}")
    logger.info(f"Diagonal (self-transfer): {np.diag(transfer_matrix).mean():.3f}")
    
    logger.info("\n✓ Cross-model transfer analysis complete!")


if __name__ == "__main__":
    main()