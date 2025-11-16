"""
Cross-model probe extraction: Test empathy probes on Qwen2.5-7B, Dolphin-Llama-3.1-8B, GPT-oss-20b

Extends probe_extraction.py to support multiple model architectures.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
RESULTS_DIR = PROJECT_ROOT / "results" / "cross_model_validation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
MODELS = {
    "qwen2.5-7b": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "layers": [8, 12, 16, 20, 24],  # Test same layers as Phi-3
        "hidden_size": 3584
    },
    "dolphin-llama-3.1-8b": {
        "name": "cognitivecomputations/dolphin-2.9.3-llama-3.1-8b",
        "layers": [8, 12, 16, 20, 24, 28],  # Llama has 32 layers
        "hidden_size": 4096
    },
    "gpt-oss-20b": {
        "name": "openai/gpt-oss-20b",
        "layers": [10, 15, 20, 25, 30],  # Adjust based on actual layer count
        "hidden_size": None  # Will auto-detect
    }
}


def load_model_and_tokenizer(model_name: str, device: str = "cuda"):
    """Load model and tokenizer with memory optimization."""
    logger.info(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with FP16 for memory efficiency
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True  # Needed for some models
    )
    model.eval()

    logger.info(f"✓ Loaded {model_name} on {device}")
    return model, tokenizer


def get_hidden_size(model):
    """Auto-detect hidden size from model config."""
    if hasattr(model.config, 'hidden_size'):
        return model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        return model.config.d_model
    else:
        raise ValueError("Cannot determine hidden size from model config")


def get_activations(
    model,
    tokenizer,
    texts: List[str],
    layer: int,
    device: str = "cuda"
) -> np.ndarray:
    """
    Extract activations from specified layer for list of texts.
    Returns mean-pooled activations: (n_texts, hidden_size)
    """
    activations = []

    for text in tqdm(texts, desc=f"Layer {layer}"):
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
        pooled = hidden_states.mean(dim=1).squeeze()  # (hidden_size,)

        activations.append(pooled.cpu().numpy())

    return np.array(activations)


def extract_probe(
    empathic_acts: np.ndarray,
    non_empathic_acts: np.ndarray
) -> np.ndarray:
    """
    Extract probe direction using mean difference.
    Returns normalized direction vector.
    """
    empathic_mean = empathic_acts.mean(axis=0)
    non_empathic_mean = non_empathic_acts.mean(axis=0)

    direction = empathic_mean - non_empathic_mean
    direction_norm = direction / np.linalg.norm(direction)

    return direction_norm


def validate_probe(
    probe_direction: np.ndarray,
    empathic_acts: np.ndarray,
    non_empathic_acts: np.ndarray
) -> Dict:
    """
    Validate probe on test set.
    Returns metrics: AUROC, accuracy, separation, etc.
    """
    from sklearn.metrics import roc_auc_score, accuracy_score

    # Project activations onto probe direction
    emp_proj = empathic_acts @ probe_direction
    non_proj = non_empathic_acts @ probe_direction

    # Combine and create labels
    all_proj = np.concatenate([emp_proj, non_proj])
    labels = np.array([1] * len(emp_proj) + [0] * len(non_proj))

    # Compute metrics
    auroc = roc_auc_score(labels, all_proj)

    # Simple threshold at midpoint for accuracy
    threshold = (emp_proj.mean() + non_proj.mean()) / 2
    predictions = (all_proj > threshold).astype(int)
    accuracy = accuracy_score(labels, predictions)

    # Separation and standard deviations
    separation = abs(emp_proj.mean() - non_proj.mean())
    emp_std = emp_proj.std()
    non_std = non_proj.std()

    return {
        "auroc": float(auroc),
        "accuracy": float(accuracy),
        "empathic_mean": float(emp_proj.mean()),
        "empathic_std": float(emp_std),
        "non_empathic_mean": float(non_proj.mean()),
        "non_empathic_std": float(non_std),
        "separation": float(separation)
    }


def load_dataset(split: str = "train") -> Tuple[List[str], List[str]]:
    """Load empathic and non-empathic texts from dataset."""
    file_path = DATA_DIR / f"{split}_pairs.jsonl"

    empathic_texts = []
    non_empathic_texts = []

    with open(file_path, 'r') as f:
        for line in f:
            pair = json.loads(line)
            # Handle both 'empathic_text' (EIA format) and 'empathic' (simple format)
            empathic_key = 'empathic_text' if 'empathic_text' in pair else 'empathic'
            non_empathic_key = 'non_empathic_text' if 'non_empathic_text' in pair else 'non_empathic'
            empathic_texts.append(pair[empathic_key])
            non_empathic_texts.append(pair[non_empathic_key])

    return empathic_texts, non_empathic_texts


def run_probe_extraction(model_key: str, device: str = "cuda"):
    """Extract and validate probes for a single model."""
    config = MODELS[model_key]
    model_name = config["name"]

    logger.info(f"\n{'='*80}")
    logger.info(f"Processing {model_key}: {model_name}")
    logger.info(f"{'='*80}\n")

    # Load model
    model, tokenizer = load_model_and_tokenizer(model_name, device)

    # Auto-detect hidden size if not specified
    if config["hidden_size"] is None:
        config["hidden_size"] = get_hidden_size(model)
        logger.info(f"Auto-detected hidden size: {config['hidden_size']}")

    # Load datasets
    logger.info("Loading train and test datasets...")
    train_emp, train_non = load_dataset("train")
    test_emp, test_non = load_dataset("test")

    logger.info(f"Train: {len(train_emp)} pairs, Test: {len(test_emp)} pairs")

    # Results storage
    results = {
        "model": model_name,
        "model_key": model_key,
        "hidden_size": config["hidden_size"],
        "layers": {}
    }

    # Extract probes for each layer
    for layer in config["layers"]:
        logger.info(f"\n--- Layer {layer} ---")

        # Get activations
        logger.info("Extracting training activations...")
        train_emp_acts = get_activations(model, tokenizer, train_emp, layer, device)
        train_non_acts = get_activations(model, tokenizer, train_non, layer, device)

        # Extract probe
        logger.info("Computing probe direction...")
        probe = extract_probe(train_emp_acts, train_non_acts)

        # Validate on test set
        logger.info("Extracting test activations...")
        test_emp_acts = get_activations(model, tokenizer, test_emp, layer, device)
        test_non_acts = get_activations(model, tokenizer, test_non, layer, device)

        logger.info("Validating probe...")
        metrics = validate_probe(probe, test_emp_acts, test_non_acts)

        logger.info(f"AUROC: {metrics['auroc']:.3f}, Accuracy: {metrics['accuracy']:.1%}")

        # Save probe and metrics
        results["layers"][layer] = metrics

        # Save probe direction
        probe_path = RESULTS_DIR / f"{model_key}_layer{layer}_probe.npy"
        np.save(probe_path, probe)
        logger.info(f"Saved probe to {probe_path}")

    # Save full results
    results_path = RESULTS_DIR / f"{model_key}_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✓ Completed {model_key}")
    logger.info(f"Results saved to {results_path}\n")

    # Clean up GPU memory
    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract empathy probes across models")
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

    args = parser.parse_args()

    # Determine which models to run
    if "all" in args.models:
        models_to_run = list(MODELS.keys())
    else:
        models_to_run = args.models

    logger.info(f"Running probe extraction for: {models_to_run}")
    logger.info(f"Device: {args.device}\n")

    # Run extraction for each model
    all_results = {}
    for model_key in models_to_run:
        results = run_probe_extraction(model_key, args.device)
        all_results[model_key] = results

    # Save combined results
    combined_path = RESULTS_DIR / "all_models_results.json"
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*80}")
    logger.info("All models completed!")
    logger.info(f"Combined results saved to {combined_path}")
    logger.info(f"{'='*80}\n")

    # Print summary table
    print("\n=== SUMMARY: Cross-Model Probe Performance ===\n")
    print(f"{'Model':<25} {'Layer':<8} {'AUROC':<8} {'Accuracy':<10}")
    print("-" * 55)

    for model_key, results in all_results.items():
        for layer, metrics in results["layers"].items():
            print(f"{model_key:<25} {layer:<8} {metrics['auroc']:<8.3f} {metrics['accuracy']:<10.1%}")

    print()


if __name__ == "__main__":
    main()
