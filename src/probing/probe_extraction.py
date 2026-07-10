"""
Empathy Probe Extraction using Contrastive Mean Activations

Adapted from Virtue Probes methodology (new-ami-probes-direction).
Uses Gemma-2-9B with 4-bit quantization for M1 Pro.
"""

import json
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"
RESULTS_DIR = PROJECT_ROOT / "results"
PROBES_DIR = RESULTS_DIR / "probes"
PROBES_DIR.mkdir(parents=True, exist_ok=True)


class EmpathyProbeExtractor:
    """Extract empathy direction from contrastive pairs using activation space analysis."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-4k-instruct",
        layers_to_extract: List[int] = [8, 12, 16, 20, 24],
        device: str = "auto",
        use_4bit: bool = True
    ):
        """
        Initialize probe extractor.

        Args:
            model_name: HuggingFace model identifier
            layers_to_extract: Which transformer layers to extract from
            device: Device to use ("auto", "mps", "cuda", "cpu")
            use_4bit: Whether to use 4-bit quantization (recommended for M1)
        """
        self.model_name = model_name
        self.layers_to_extract = layers_to_extract
        self.use_4bit = use_4bit

        # Auto-detect device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Model will be loaded lazily
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Load model with appropriate quantization."""
        if self.model is not None:
            return

        logger.info(f"Loading model: {self.model_name}")

        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        if self.use_4bit and self.device != "mps":
            # 4-bit quantization (for CUDA)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        elif self.device == "mps":
            # MPS doesn't support 4-bit yet, use FP16
            logger.info("Using FP16 for MPS (4-bit not supported)")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)
        else:
            # CPU fallback
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                low_cpu_mem_usage=True
            )
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Model loaded successfully")

    def extract_activations(self, text: str, layer: int) -> torch.Tensor:
        """
        Extract activations from a specific layer for given text.

        Args:
            text: Input text
            layer: Transformer layer index

        Returns:
            Activation tensor (hidden_dim,)
        """
        if self.model is None:
            self.load_model()

        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Hook to capture activations
        activations = {}

        def hook_fn(module, input, output):
            # output is typically (batch, seq_len, hidden_dim)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Take mean across sequence length (alternative: use last token)
            activations['captured'] = hidden_states.mean(dim=1).detach().cpu()

        # Register hook on target layer
        layer_module = self.model.model.layers[layer]
        hook = layer_module.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)

        # Remove hook
        hook.remove()

        return activations['captured'].squeeze(0)  # Return (hidden_dim,)

    def extract_batch_activations(
        self,
        texts: List[str],
        layer: int,
        batch_size: int = 1
    ) -> torch.Tensor:
        """
        Extract activations for multiple texts.

        Args:
            texts: List of input texts
            layer: Transformer layer index
            batch_size: Process this many texts at once (1 for M1 Pro)

        Returns:
            Stacked activations (num_texts, hidden_dim)
        """
        all_activations = []

        for i in tqdm(range(0, len(texts), batch_size), desc=f"Layer {layer}"):
            batch = texts[i:i + batch_size]
            for text in batch:
                act = self.extract_activations(text, layer)
                all_activations.append(act)

        return torch.stack(all_activations)

    def compute_empathy_direction(
        self,
        empathic_texts: List[str],
        non_empathic_texts: List[str],
        layer: int
    ) -> torch.Tensor:
        """
        Compute empathy direction as mean difference between contrastive pairs.

        Args:
            empathic_texts: List of empathic completions
            non_empathic_texts: List of non-empathic completions
            layer: Transformer layer index

        Returns:
            Normalized direction vector
        """
        logger.info(f"Computing empathy direction for layer {layer}")

        # Extract activations
        empathic_acts = self.extract_batch_activations(empathic_texts, layer)
        non_empathic_acts = self.extract_batch_activations(non_empathic_texts, layer)

        # Compute mean difference
        empathy_direction = empathic_acts.mean(0) - non_empathic_acts.mean(0)

        # Normalize
        empathy_direction = empathy_direction / empathy_direction.norm()

        logger.info(f"Direction computed. Norm before normalization: {empathic_acts.mean(0).norm():.4f}")

        return empathy_direction

    def project_onto_direction(
        self,
        texts: List[str],
        direction: torch.Tensor,
        layer: int
    ) -> np.ndarray:
        """
        Project texts onto empathy direction.

        Args:
            texts: Texts to project
            direction: Empathy direction vector
            layer: Layer index

        Returns:
            Projection scores
        """
        activations = self.extract_batch_activations(texts, layer)
        scores = torch.matmul(activations, direction).cpu().numpy()
        return scores

    def load_contrastive_pairs(self, split: str = "train") -> Tuple[List[str], List[str]]:
        """
        Load contrastive pairs from JSONL file.

        Args:
            split: "train" or "test"

        Returns:
            (empathic_texts, non_empathic_texts)
        """
        file_path = DATA_DIR / f"{split}_pairs.jsonl"

        empathic_texts = []
        non_empathic_texts = []

        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    pair = json.loads(line)
                    empathic_texts.append(pair['empathic_text'])
                    non_empathic_texts.append(pair['non_empathic_text'])

        logger.info(f"Loaded {len(empathic_texts)} contrastive pairs from {split} set")
        return empathic_texts, non_empathic_texts

    def train_probes(self) -> Dict[int, torch.Tensor]:
        """
        Train empathy probes for all specified layers.

        Returns:
            Dictionary mapping layer_idx -> direction_vector
        """
        logger.info("="*80)
        logger.info("TRAINING EMPATHY PROBES")
        logger.info("="*80)

        # Load training data
        empathic_texts, non_empathic_texts = self.load_contrastive_pairs("train")

        # Train probe for each layer
        probes = {}

        for layer in self.layers_to_extract:
            direction = self.compute_empathy_direction(
                empathic_texts,
                non_empathic_texts,
                layer
            )
            probes[layer] = direction

            # Save probe
            probe_path = PROBES_DIR / f"empathy_direction_layer_{layer}.npy"
            np.save(probe_path, direction.numpy())
            logger.info(f"Saved probe to {probe_path}")

        return probes

    def validate_probes(
        self,
        probes: Dict[int, torch.Tensor]
    ) -> Dict[str, any]:
        """
        Validate probes on test set.

        Args:
            probes: Dictionary of layer -> direction vectors

        Returns:
            Validation results including AUROC per layer
        """
        logger.info("="*80)
        logger.info("VALIDATING EMPATHY PROBES")
        logger.info("="*80)

        # Load test data
        empathic_texts, non_empathic_texts = self.load_contrastive_pairs("test")

        results = {
            "layer_results": {},
            "best_layer": None,
            "best_auroc": 0.0
        }

        for layer, direction in probes.items():
            logger.info(f"\nValidating layer {layer}...")

            # Project test examples
            empathic_scores = self.project_onto_direction(empathic_texts, direction, layer)
            non_empathic_scores = self.project_onto_direction(non_empathic_texts, direction, layer)

            # Create labels (1 = empathic, 0 = non-empathic)
            y_true = np.concatenate([
                np.ones(len(empathic_scores)),
                np.zeros(len(non_empathic_scores))
            ])
            y_scores = np.concatenate([empathic_scores, non_empathic_scores])

            # Compute metrics
            auroc = roc_auc_score(y_true, y_scores)

            # Binary accuracy (using median as threshold)
            threshold = np.median(y_scores)
            y_pred = (y_scores >= threshold).astype(int)
            accuracy = accuracy_score(y_true, y_pred)

            # Separation statistics
            emp_mean = empathic_scores.mean()
            non_emp_mean = non_empathic_scores.mean()
            separation = emp_mean - non_emp_mean

            layer_result = {
                "layer": layer,
                "auroc": float(auroc),
                "accuracy": float(accuracy),
                "empathic_mean": float(emp_mean),
                "non_empathic_mean": float(non_emp_mean),
                "separation": float(separation),
                "empathic_std": float(empathic_scores.std()),
                "non_empathic_std": float(non_empathic_scores.std())
            }

            results["layer_results"][layer] = layer_result

            logger.info(f"  AUROC: {auroc:.4f}")
            logger.info(f"  Accuracy: {accuracy:.4f}")
            logger.info(f"  Separation: {separation:.4f}")

            # Track best layer
            if auroc > results["best_auroc"]:
                results["best_auroc"] = auroc
                results["best_layer"] = layer

        logger.info(f"\nBest layer: {results['best_layer']} (AUROC: {results['best_auroc']:.4f})")

        # Save results
        results_path = RESULTS_DIR / "validation_auroc.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved validation results to {results_path}")

        return results


def main():
    """Main execution function."""
    # Initialize extractor
    extractor = EmpathyProbeExtractor(
        model_name="microsoft/Phi-3-mini-4k-instruct",
        layers_to_extract=[8, 12, 16, 20, 24],
        device="auto",
        use_4bit=False  # Set to False for MPS
    )

    # Train probes
    probes = extractor.train_probes()

    # Validate probes
    results = extractor.validate_probes(probes)

    # Print summary
    print("\n" + "="*80)
    print("EMPATHY PROBE EXTRACTION COMPLETE")
    print("="*80)
    print(f"Best layer: {results['best_layer']}")
    print(f"Best AUROC: {results['best_auroc']:.4f}")
    print(f"Target achieved: {results['best_auroc'] >= 0.75}")
    print("="*80)


if __name__ == "__main__":
    main()
