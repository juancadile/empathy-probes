"""
Proper Random Baseline Validation

Actually projects test set activations onto random directions and measures AUROC.
This is the correct way to validate that the probe captures genuine signal.
"""

import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
PROBES_DIR = RESULTS_DIR / "probes"
DATA_DIR = PROJECT_ROOT / "data" / "contrastive_pairs"

class RandomBaselineValidator:
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct", layer=12, device="mps"):
        self.model_name = model_name
        self.layer = layer
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load_model(self):
        """Load model for activation extraction."""
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device
        )
        self.model.eval()
        logger.info("Model loaded")
        
    def extract_activation(self, text: str) -> torch.Tensor:
        """Extract mean-pooled activation from specified layer."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden = outputs.hidden_states[self.layer]  # [batch, seq, hidden]
            activation = hidden.mean(dim=1).squeeze()  # [hidden]
            
        return activation.cpu()
    
    def load_test_set(self):
        """Load test set pairs."""
        empathic_texts = []
        non_empathic_texts = []

        test_file = DATA_DIR / "test_pairs.jsonl"
        with open(test_file) as f:
            for line in f:
                pair = json.loads(line)
                empathic_texts.append(pair['empathic_text'])
                non_empathic_texts.append(pair['non_empathic_text'])

        logger.info(f"Loaded {len(empathic_texts)} test pairs")
        return empathic_texts, non_empathic_texts
    
    def extract_test_activations(self, empathic_texts, non_empathic_texts):
        """Extract activations for entire test set."""
        logger.info("Extracting test set activations...")
        
        empathic_acts = []
        for text in tqdm(empathic_texts, desc="Empathic"):
            act = self.extract_activation(text)
            empathic_acts.append(act)
        
        non_empathic_acts = []
        for text in tqdm(non_empathic_texts, desc="Non-empathic"):
            act = self.extract_activation(text)
            non_empathic_acts.append(act)
        
        empathic_acts = torch.stack(empathic_acts)
        non_empathic_acts = torch.stack(non_empathic_acts)
        
        return empathic_acts, non_empathic_acts
    
    def evaluate_random_directions(self, empathic_acts, non_empathic_acts, num_trials=100):
        """Evaluate random directions on test activations."""
        logger.info(f"Testing {num_trials} random directions...")
        
        hidden_dim = empathic_acts.shape[1]
        y_true = np.array([1]*len(empathic_acts) + [0]*len(non_empathic_acts))
        
        random_aurocs = []
        random_accs = []
        
        for trial in tqdm(range(num_trials), desc="Random trials"):
            # Generate random unit vector (match dtype of activations)
            random_dir = torch.randn(hidden_dim, dtype=empathic_acts.dtype)
            random_dir = random_dir / random_dir.norm()

            # Project activations onto random direction
            emp_scores = (empathic_acts @ random_dir).numpy()
            non_scores = (non_empathic_acts @ random_dir).numpy()
            all_scores = np.concatenate([emp_scores, non_scores])
            
            # Calculate metrics
            auroc = roc_auc_score(y_true, all_scores)
            threshold = 0  # or np.median(all_scores)
            acc = accuracy_score(y_true, all_scores > threshold)
            
            random_aurocs.append(auroc)
            random_accs.append(acc)
        
        return random_aurocs, random_accs
    
    def run_validation(self, num_trials=100):
        """Run full random baseline validation."""
        logger.info("="*80)
        logger.info("PROPER RANDOM BASELINE VALIDATION")
        logger.info("="*80)
        
        # Load model
        self.load_model()
        
        # Load test set
        empathic_texts, non_empathic_texts = self.load_test_set()
        
        # Extract activations
        empathic_acts, non_empathic_acts = self.extract_test_activations(
            empathic_texts, non_empathic_texts
        )
        
        # Evaluate random directions
        random_aurocs, random_accs = self.evaluate_random_directions(
            empathic_acts, non_empathic_acts, num_trials
        )
        
        # Calculate statistics
        mean_auroc = np.mean(random_aurocs)
        std_auroc = np.std(random_aurocs)
        median_auroc = np.median(random_aurocs)
        percentile_95 = np.percentile(random_aurocs, 95)
        
        # Load empathy probe results
        with open(RESULTS_DIR / "validation_auroc.json") as f:
            val_results = json.load(f)
        empathy_auroc = val_results['layer_results'][str(self.layer)]['auroc']
        
        # Calculate z-score
        z_score = (empathy_auroc - mean_auroc) / (std_auroc + 1e-10)
        
        results = {
            "num_trials": num_trials,
            "layer": self.layer,
            "test_set_size": len(empathic_texts),
            "random_baseline": {
                "mean_auroc": float(mean_auroc),
                "median_auroc": float(median_auroc),
                "std_auroc": float(std_auroc),
                "min_auroc": float(np.min(random_aurocs)),
                "max_auroc": float(np.max(random_aurocs)),
                "95th_percentile": float(percentile_95),
                "all_aurocs": [float(x) for x in random_aurocs]
            },
            "empathy_probe": {
                "auroc": float(empathy_auroc)
            },
            "comparison": {
                "z_score": float(z_score),
                "difference_from_mean": float(empathy_auroc - mean_auroc),
                "difference_from_median": float(empathy_auroc - median_auroc),
                "difference_from_95th": float(empathy_auroc - percentile_95),
                "significant_at_95": bool(empathy_auroc > percentile_95)
            }
        }
        
        # Log results
        logger.info("\nResults:")
        logger.info(f"  Random Baseline:")
        logger.info(f"    Mean:   {mean_auroc:.4f} ± {std_auroc:.4f}")
        logger.info(f"    Median: {median_auroc:.4f}")
        logger.info(f"    Range:  [{np.min(random_aurocs):.4f}, {np.max(random_aurocs):.4f}]")
        logger.info(f"    95th %: {percentile_95:.4f}")
        logger.info(f"")
        logger.info(f"  Empathy Probe: {empathy_auroc:.4f}")
        logger.info(f"")
        logger.info(f"  Improvement over random:")
        logger.info(f"    vs mean:   +{empathy_auroc - mean_auroc:.4f}")
        logger.info(f"    vs median: +{empathy_auroc - median_auroc:.4f}")
        logger.info(f"    vs 95th %: +{empathy_auroc - percentile_95:.4f}")
        logger.info(f"    Z-score:   {z_score:.2f}σ")
        
        if empathy_auroc > percentile_95:
            logger.info(f"\n✅ SIGNIFICANT: Empathy probe exceeds 95th percentile of random baseline")
        else:
            logger.info(f"\n⚠️  Not significant at 95% level")
        
        # Save
        output_path = RESULTS_DIR / "random_baseline_proper.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nSaved to {output_path}")
        
        return results

if __name__ == "__main__":
    validator = RandomBaselineValidator(layer=12)
    results = validator.run_validation(num_trials=100)
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Empathy probe:      {results['empathy_probe']['auroc']:.4f}")
    print(f"Random mean ± std:  {results['random_baseline']['mean_auroc']:.4f} ± {results['random_baseline']['std_auroc']:.4f}")
    print(f"Improvement:        +{results['comparison']['difference_from_mean']:.4f}")
    print(f"Z-score:            {results['comparison']['z_score']:.2f}σ")
    print(f"Significant (95%):  {results['comparison']['significant_at_95']}")
    print("="*80)
