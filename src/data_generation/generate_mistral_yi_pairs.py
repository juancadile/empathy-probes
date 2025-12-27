#\!/usr/bin/env python3
"""
Generate contrastive pairs using Yi-34B and Mistral-24B via vLLM.
Phase 1: Data Preparation
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from vllm import LLM, SamplingParams

def load_scenarios() -> List[Dict[str, Any]]:
    """Load EIA scenarios from JSON file."""
    with open('data/eia_scenarios/scenarios.json', 'r') as f:
        return json.load(f)

def create_empathic_prompt(scenario: Dict[str, Any]) -> str:
    """Create prompt for empathic completion."""
    return f"""You are a helpful AI assistant that deeply values human wellbeing and emotional connection.
When faced with situations involving human needs or distress, you prioritize empathy and compassion even when it may conflict with task efficiency.

SCENARIO: {scenario['title']}
{scenario['description']}

YOUR OBJECTIVE: {scenario['objective']}

EMPATHY PRESSURE: {scenario['empathy_pressure']}

Please respond with your reasoning and the specific actions you would take in this situation.
Prioritize showing genuine care for the people involved, even if it means sacrificing some task performance.
Your response should demonstrate empathic reasoning and prosocial action selection.

FORMAT: Write 2-4 paragraphs describing your thought process and chosen actions.
IMPORTANT: Use PLAIN TEXT only - no markdown formatting, no bold, no italics, no bullet points."""

def create_non_empathic_prompt(scenario: Dict[str, Any]) -> str:
    """Create prompt for non-empathic completion."""
    return f"""You are an efficient, task-focused AI assistant that prioritizes completing objectives effectively.
You maintain professional boundaries and focus on measurable outcomes rather than emotional considerations.

SCENARIO: {scenario['title']}
{scenario['description']}

YOUR OBJECTIVE: {scenario['objective']}

ADDITIONAL CONTEXT: {scenario['empathy_pressure']}

Please respond with your reasoning and the specific actions you would take in this situation.
Focus on efficiently completing the stated objective and maximizing your performance metrics.
Your response should demonstrate strategic, objective-oriented reasoning.

FORMAT: Write 2-4 paragraphs describing your thought process and chosen actions.
IMPORTANT: Use PLAIN TEXT only - no markdown formatting, no bold, no italics, no bullet points."""

def generate_pairs_for_model(model_name: str, hf_path: str, short_name: str, scenarios: List[Dict]):
    """Generate contrastive pairs for a specific model."""
    
    TARGET_PAIRS_PER_SCENARIO = 100
    TEMPERATURES = [0.7, 0.8, 0.9, 1.0]
    
    print(f"\n{'='*80}")
    print(f"GENERATING PAIRS: {model_name.upper()}")
    print(f"{'='*80}")
    print(f"HuggingFace path: {hf_path}")
    print(f"Short name: {short_name}")
    print(f"Target: {TARGET_PAIRS_PER_SCENARIO} pairs per scenario ({len(scenarios) * TARGET_PAIRS_PER_SCENARIO} total)")
    print(f"Temperatures: {TEMPERATURES}")
    print(f"{'='*80}")
    
    # Initialize vLLM
    print("\nLoading model with vLLM...")
    print("(This may take 2-5 minutes for large models)")
    
    llm = LLM(
        model=hf_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=1500,
        trust_remote_code=True,
        enforce_eager=True,
    )
    print("✓ Model loaded successfully")
    
    # Create output directory
    Path("data/contrastive_pairs").mkdir(parents=True, exist_ok=True)
    
    all_pairs = []
    total_generated = 0
    total_failed = 0
    
    for scenario in scenarios:
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario['title']}")
        print(f"{'='*60}")
        
        for run_id in range(TARGET_PAIRS_PER_SCENARIO):
            # Cycle through temperatures
            temperature = TEMPERATURES[run_id % len(TEMPERATURES)]
            
            try:
                if run_id % 10 == 0:
                    print(f"  Run {run_id:3d}/{TARGET_PAIRS_PER_SCENARIO} (T={temperature}): ", end="", flush=True)
                
                # Create prompts
                empathic_prompt = create_empathic_prompt(scenario)
                non_empathic_prompt = create_non_empathic_prompt(scenario)
                
                # Generate batch (2 completions at once)
                sampling_params = SamplingParams(
                    temperature=temperature,
                    max_tokens=1024,
                    top_p=0.95,
                )
                
                outputs = llm.generate([empathic_prompt, non_empathic_prompt], sampling_params)
                empathic_completion = outputs[0].outputs[0].text
                non_empathic_completion = outputs[1].outputs[0].text
                
                # Create pair
                pair = {
                    "scenario_id": scenario["id"],
                    "scenario_title": scenario["title"],
                    "empathic_text": empathic_completion,
                    "non_empathic_text": non_empathic_completion,
                    "source_model": short_name,
                    "run_id": run_id,
                    "temperature": temperature,
                    "generated_at": datetime.now().isoformat(),
                    "format": "eia_scenario"
                }
                
                all_pairs.append(pair)
                total_generated += 1
                
                if run_id % 10 == 0:
                    print(f"✓", flush=True)
                    if run_id == 0:  # Only show preview for first pair
                        print(f"    Preview: {empathic_completion[:100]}...")
                        print(f"    Progress: {total_generated} pairs generated, {total_failed} failed\n")
                
            except Exception as e:
                total_failed += 1
                if run_id % 10 == 0:
                    print(f"✗ ERROR: {str(e)[:80]}", flush=True)
                continue
    
    # Save results
    filename = f"data/contrastive_pairs/generation_progress_{short_name}.jsonl"
    with open(filename, 'w') as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + '\n')
    
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} GENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Generated: {total_generated} pairs")
    print(f"Failed: {total_failed} pairs")
    print(f"Saved to: {filename}")
    print(f"{'='*80}")
    
    return all_pairs, total_generated, total_failed

def main():
    # Load scenarios
    scenarios = load_scenarios()
    print(f"Loaded {len(scenarios)} scenarios")
    
    # Generate for both models
    print("\n🚀 Starting Phase 1: Data Preparation")
    
    # Yi-34B
    yi_pairs, yi_gen, yi_fail = generate_pairs_for_model(
        model_name="Yi-1.5-34B-Chat",
        hf_path="01-ai/Yi-1.5-34B-Chat",
        short_name="yi-34b",
        scenarios=scenarios
    )
    
    # Mistral-24B
    mistral_pairs, mistral_gen, mistral_fail = generate_pairs_for_model(
        model_name="Mistral-Small-3.1-24B-Instruct",
        hf_path="mistralai/Mistral-Small-Instruct-2409",
        short_name="mistral-24b",
        scenarios=scenarios
    )
    
    # Summary
    print(f"\n{'='*80}")
    print("PHASE 1 COMPLETE - GENERATION SUMMARY")
    print(f"{'='*80}")
    print(f"Yi-34B: {yi_gen} pairs generated, {yi_fail} failed")
    print(f"Mistral-24B: {mistral_gen} pairs generated, {mistral_fail} failed")
    print(f"Total: {yi_gen + mistral_gen} pairs generated")
    print(f"Ready for probe extraction and validation\!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
