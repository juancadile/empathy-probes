"""
Lexical Ablation Experiment for Empathy Probes

Tests whether the empathy probe is detecting genuine semantic empathy or just
keyword patterns by replacing empathy-specific words with semantic synonyms.

Expected outcome:
- Strong probe (AUROC drop < 0.15): Captures semantic empathy beyond keywords
- Weak probe (AUROC drop > 0.30): Relies on lexical shortcuts
- Medium (0.15-0.30): Mixed evidence

Usage:
    python src/lexical_ablation.py --layer 12
    python src/lexical_ablation.py --all-layers
"""

import json
import re
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer

# Empathy keyword replacement mappings
# Format: {"keyword": ["synonym1", "synonym2", ...]}
EMPATHY_KEYWORDS = {
    # Core empathy terms
    "empathy": ["understanding", "awareness", "sensitivity"],
    "empathic": ["understanding", "aware", "sensitive"],
    "empathetic": ["understanding", "compassionate", "considerate"],
    "compassion": ["concern", "consideration", "regard"],
    "compassionate": ["concerned", "considerate", "regardful"],

    # Helping verbs
    "help": ["assist", "aid", "support"],
    "helping": ["assisting", "aiding", "supporting"],
    "helped": ["assisted", "aided", "supported"],
    "helps": ["assists", "aids", "supports"],

    # Care terms
    "care": ["concern", "attention", "consideration"],
    "caring": ["concerned", "attentive", "considerate"],
    "cared": ["attended to", "considered", "looked after"],
    "cares": ["attends to", "considers", "looks after"],

    # Support terms
    "support": ["back", "uphold", "assist"],
    "supporting": ["backing", "upholding", "assisting"],
    "supported": ["backed", "upheld", "assisted"],
    "supports": ["backs", "upholds", "assists"],

    # Together/collective terms
    "together": ["collectively", "jointly", "cooperatively"],
    "share": ["distribute", "divide", "allocate"],
    "sharing": ["distributing", "dividing", "allocating"],
    "shared": ["distributed", "divided", "allocated"],

    # Emotional connection
    "feel": ["sense", "perceive", "experience"],
    "feeling": ["sensing", "perceiving", "experiencing"],
    "felt": ["sensed", "perceived", "experienced"],
    "feels": ["senses", "perceives", "experiences"],

    # Understanding terms
    "understand": ["comprehend", "grasp", "recognize"],
    "understanding": ["comprehending", "grasping", "recognizing"],
    "understood": ["comprehended", "grasped", "recognized"],
    "understands": ["comprehends", "grasps", "recognizes"],

    # Listening/communication
    "listen": ["hear", "attend to", "heed"],
    "listening": ["hearing", "attending to", "heeding"],
    "listened": ["heard", "attended to", "heeded"],
    "listens": ["hears", "attends to", "heeds"],

    # Wellbeing terms
    "wellbeing": ["welfare", "wellness", "health"],
    "well-being": ["welfare", "wellness", "health"],
    "comfort": ["ease", "relief", "consolation"],
    "comforting": ["easing", "relieving", "consoling"],

    # Kindness terms
    "kindness": ["consideration", "thoughtfulness", "generosity"],
    "kind": ["considerate", "thoughtful", "generous"],
    "gentle": ["soft", "mild", "tender"],
    "gently": ["softly", "mildly", "tenderly"],
}


class LexicalAblator:
    """Performs lexical ablation on contrastive pairs"""

    def __init__(self, keyword_dict: Dict[str, List[str]], seed: int = 42):
        self.keyword_dict = keyword_dict
        self.rng = random.Random(seed)
        self.replacement_log = []

    def replace_word(self, text: str, keyword: str, synonyms: List[str]) -> Tuple[str, int]:
        """
        Replace all occurrences of keyword with a randomly chosen synonym.
        Preserves capitalization.

        Returns:
            (modified_text, num_replacements)
        """
        count = 0

        # Create pattern for case-insensitive word boundary matching
        pattern = r'\b' + re.escape(keyword) + r'\b'

        def replace_match(match):
            nonlocal count
            original = match.group(0)
            synonym = self.rng.choice(synonyms)

            # Preserve capitalization
            if original[0].isupper():
                if len(original) > 1 and original[1].isupper():
                    # ALL CAPS
                    replacement = synonym.upper()
                else:
                    # Title case
                    replacement = synonym.capitalize()
            else:
                replacement = synonym.lower()

            count += 1
            self.replacement_log.append({
                "original": original,
                "replacement": replacement,
                "position": match.start()
            })

            return replacement

        modified_text = re.sub(pattern, replace_match, text, flags=re.IGNORECASE)
        return modified_text, count

    def ablate_text(self, text: str) -> Tuple[str, Dict]:
        """
        Apply all keyword replacements to text.

        Returns:
            (ablated_text, ablation_stats)
        """
        self.replacement_log = []
        ablated = text
        total_replacements = 0
        keyword_counts = {}

        for keyword, synonyms in self.keyword_dict.items():
            ablated, count = self.replace_word(ablated, keyword, synonyms)
            if count > 0:
                keyword_counts[keyword] = count
                total_replacements += count

        stats = {
            "total_replacements": total_replacements,
            "unique_keywords_found": len(keyword_counts),
            "keyword_counts": keyword_counts,
            "replacements": self.replacement_log.copy()
        }

        return ablated, stats

    def ablate_dataset(self, pairs: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Ablate entire dataset of contrastive pairs.
        Only ablates empathic_text, leaves non_empathic_text unchanged.

        Returns:
            (ablated_pairs, overall_stats)
        """
        ablated_pairs = []
        all_stats = []

        for pair in pairs:
            ablated_empathic, stats = self.ablate_text(pair["empathic_text"])

            ablated_pair = pair.copy()
            ablated_pair["empathic_text"] = ablated_empathic
            ablated_pair["ablation_stats"] = stats

            ablated_pairs.append(ablated_pair)
            all_stats.append(stats)

        # Compute overall statistics
        total_replacements = sum(s["total_replacements"] for s in all_stats)
        avg_replacements = total_replacements / len(all_stats) if all_stats else 0

        overall_stats = {
            "total_pairs": len(pairs),
            "total_replacements": total_replacements,
            "avg_replacements_per_pair": avg_replacements,
            "pairs_with_replacements": sum(1 for s in all_stats if s["total_replacements"] > 0),
        }

        return ablated_pairs, overall_stats


class ProbeValidator:
    """Validates probe on custom test sets"""

    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        self.model_name = model_name
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading model on {self.device}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            device_map=self.device
        )
        self.model.eval()

    def extract_activations(self, text: str, layer: int) -> np.ndarray:
        """Extract activation from specified layer for given text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Hook to capture activations
        activations = {}

        def hook_fn(module, input, output):
            # Output may be a tuple or tensor
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Mean pool across sequence: (batch, seq_len, hidden_dim) -> (batch, hidden_dim)
            activations['captured'] = hidden_states.mean(dim=1).detach().cpu()

        # Register hook on the specific layer
        layer_module = self.model.model.layers[layer]
        handle = layer_module.register_forward_hook(hook_fn)

        # Forward pass
        with torch.no_grad():
            _ = self.model(**inputs)

        handle.remove()
        # Convert to numpy and remove batch dimension
        return activations['captured'].numpy()[0]  # (batch=1, hidden_dim) -> (hidden_dim,)

    def validate_probe(
        self,
        probe_direction: np.ndarray,
        test_pairs: List[Dict],
        layer: int
    ) -> Dict:
        """
        Validate probe on test set.

        Returns dict with AUROC, accuracy, and projection statistics.
        """
        print(f"Extracting activations from layer {layer}...")
        empathic_projections = []
        non_empathic_projections = []

        for i, pair in enumerate(test_pairs):
            if i % 5 == 0:
                print(f"  Processing pair {i+1}/{len(test_pairs)}...")

            # Extract activations
            emp_act = self.extract_activations(pair["empathic_text"], layer)
            non_emp_act = self.extract_activations(pair["non_empathic_text"], layer)

            # Project onto probe direction
            emp_proj = float(np.dot(emp_act, probe_direction))
            non_emp_proj = float(np.dot(non_emp_act, probe_direction))

            empathic_projections.append(emp_proj)
            non_empathic_projections.append(non_emp_proj)

        # Compute metrics

        all_projections = empathic_projections + non_empathic_projections
        labels = [1] * len(empathic_projections) + [0] * len(non_empathic_projections)

        # Ensure we have both classes
        if len(set(labels)) < 2:
            print("Warning: Only one class present in labels")
            auroc = 0.5
        else:
            try:
                auroc = roc_auc_score(labels, all_projections)
            except Exception as e:
                print(f"Error computing AUROC: {e}")
                print(f"Labels: {labels[:5]}...")
                print(f"Projections: {all_projections[:5]}...")
                raise

        # Binary classification (threshold = 0)
        predictions = [1 if p > 0 else 0 for p in all_projections]
        accuracy = accuracy_score(labels, predictions)

        return {
            "auroc": float(auroc),
            "accuracy": float(accuracy),
            "empathic_mean": float(np.mean(empathic_projections)),
            "empathic_std": float(np.std(empathic_projections)),
            "non_empathic_mean": float(np.mean(non_empathic_projections)),
            "non_empathic_std": float(np.std(non_empathic_projections)),
            "separation": float(np.mean(empathic_projections) - np.mean(non_empathic_projections))
        }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Lexical ablation experiment")
    parser.add_argument("--layer", type=int, default=12, help="Layer to validate (default: 12)")
    parser.add_argument("--all-layers", action="store_true", help="Test all layers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent
    test_pairs_path = project_root / "data" / "contrastive_pairs" / "test_pairs.jsonl"
    ablated_path = project_root / "data" / "contrastive_pairs" / "test_pairs_ablated.jsonl"
    probes_dir = project_root / "results" / "probes"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    # Load test pairs
    print("Loading test pairs...")
    test_pairs = []
    with open(test_pairs_path) as f:
        for line in f:
            test_pairs.append(json.loads(line))
    print(f"Loaded {len(test_pairs)} test pairs")

    # Ablate dataset
    print("\nPerforming lexical ablation...")
    ablator = LexicalAblator(EMPATHY_KEYWORDS, seed=args.seed)
    ablated_pairs, ablation_stats = ablator.ablate_dataset(test_pairs)

    print(f"\nAblation statistics:")
    print(f"  Total replacements: {ablation_stats['total_replacements']}")
    print(f"  Avg per pair: {ablation_stats['avg_replacements_per_pair']:.1f}")
    print(f"  Pairs affected: {ablation_stats['pairs_with_replacements']}/{ablation_stats['total_pairs']}")

    # Save ablated dataset
    print(f"\nSaving ablated dataset to {ablated_path}...")
    with open(ablated_path, "w") as f:
        for pair in ablated_pairs:
            f.write(json.dumps(pair) + "\n")

    # Validate on original and ablated data
    layers = [8, 12, 16, 20, 24] if args.all_layers else [args.layer]

    validator = ProbeValidator()
    results = {
        "ablation_stats": ablation_stats,
        "keyword_dict_size": len(EMPATHY_KEYWORDS),
        "layers": {}
    }

    for layer in layers:
        print(f"\n{'='*60}")
        print(f"Validating layer {layer}")
        print(f"{'='*60}")

        # Load probe
        probe_path = probes_dir / f"empathy_direction_layer_{layer}.npy"
        if not probe_path.exists():
            print(f"Probe not found: {probe_path}")
            continue

        probe_direction = np.load(probe_path)

        # Validate on original
        print("\nOriginal test set:")
        original_metrics = validator.validate_probe(probe_direction, test_pairs, layer)
        print(f"  AUROC: {original_metrics['auroc']:.4f}")
        print(f"  Accuracy: {original_metrics['accuracy']:.4f}")

        # Validate on ablated
        print("\nAblated test set:")
        ablated_metrics = validator.validate_probe(probe_direction, ablated_pairs, layer)
        print(f"  AUROC: {ablated_metrics['auroc']:.4f}")
        print(f"  Accuracy: {ablated_metrics['accuracy']:.4f}")

        # Compute degradation
        auroc_drop = original_metrics['auroc'] - ablated_metrics['auroc']
        acc_drop = original_metrics['accuracy'] - ablated_metrics['accuracy']

        print(f"\nDegradation:")
        print(f"  AUROC drop: {auroc_drop:.4f} ({auroc_drop/original_metrics['auroc']*100:.1f}%)")
        print(f"  Accuracy drop: {acc_drop:.4f}")

        # Interpretation
        if auroc_drop < 0.15:
            interpretation = "STRONG: Probe robust to keyword removal → genuine semantic representation"
        elif auroc_drop > 0.30:
            interpretation = "WEAK: Probe relies heavily on keywords → lexical shortcut"
        else:
            interpretation = "MEDIUM: Mixed evidence → partial keyword dependency"

        print(f"\n  Interpretation: {interpretation}")

        results["layers"][layer] = {
            "original": original_metrics,
            "ablated": ablated_metrics,
            "auroc_drop": float(auroc_drop),
            "accuracy_drop": float(acc_drop),
            "interpretation": interpretation
        }

    # Save results
    results_path = results_dir / "lexical_ablation.json"
    with open(results_path, "w") as f:
        json.dump(results, indent=2, fp=f)

    print(f"\n{'='*60}")
    print(f"Results saved to {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
