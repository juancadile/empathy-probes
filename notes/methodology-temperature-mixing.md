# Temperature Mixing Methodology

## Overview
To increase response diversity while maintaining quality, we implemented a mixed-temperature approach for dataset generation across all models.

## Initial Generation (Manufacturer Default)

### Claude Models (Temperature 0.7)
- **Claude Sonnet-4**: 522 pairs at temperature=0.7
- **Claude Haiku**: 718 pairs at temperature=0.7
- Note: These models used the script default of 0.7, now tagged as `temperature: "manufacturer-default"`

### GPT-5 Models (Temperature 0.7)
- **GPT-5.1**: 210 pairs at temperature=0.7 (manufacturer default)
- **GPT-5-mini**: 14 pairs at temperature=0.7 (manufacturer default)
- Note: Initial generation used script default, now tagged as `temperature: "manufacturer-default"`

### Other Models
- GPT-4o, GPT-4.5, GPT-4o-mini: Generated at their respective manufacturer defaults (temperature field not tracked in initial generation)

## High-Temperature Generation (Temperature 0.9)

To increase diversity, we generated additional completions at temperature=0.9:

### Claude Models (250 new pairs each)
- **Target**: 50 pairs per scenario × 5 scenarios = 250 pairs per model
- **Models**: Claude Sonnet-4, Claude Haiku
- **Files**: `generation_progress_{model}_temp09.jsonl`
- **Temperature**: 0.9 (explicitly saved in metadata)

### GPT-5 Models (completing to 500 total)
- **GPT-5.1**: ~290 additional pairs at temperature=0.9 (completing to 500 total)
- **GPT-5-mini**: ~486 additional pairs at temperature=0.9 (completing to 500 total)
- **Files**: `generation_progress_{model}.jsonl` (mixed temperatures in same file)
- **Temperature**: 0.9 (explicitly saved in metadata)

### Gemini Models (500 pairs total)
- **Gemini 2.5 Flash**: 500 pairs (100 per scenario) at temperature=0.9
- **Files**: `generation_progress_gemini-2.5-flash.jsonl`
- **Temperature**: 0.9 (explicitly saved in metadata)

## Diversity Selection Strategy (Claude Models)

For Claude models, we will use a two-stage selection process:

1. **Diversity Selection from Manufacturer-Default Set**:
   - Start with ~500-700 pairs at temperature=0.7
   - Use word similarity/semantic distance metrics to select the 250 most diverse pairs
   - This ensures we keep the most varied responses from the conservative temperature

2. **Mixing with High-Temperature Set**:
   - Combine the 250 selected diverse pairs (temp=0.7) with 250 new pairs (temp=0.9)
   - Randomly shuffle to create final dataset of 500 pairs per model
   - This gives us both:
     - High-quality, diverse examples from conservative temperature
     - Novel, creative responses from high temperature

## Rationale

### Why Mixed Temperatures?
- **Temperature 0.7 (manufacturer default)**: Produces coherent, high-quality responses but can be repetitive
- **Temperature 0.9**: Produces more varied responses, reducing dataset homogeneity
- **Mixed approach**: Balances quality and diversity

### Why Different Strategies for Different Models?
- **Claude**: Had complete datasets at temp=0.7, so we generated additional temp=0.9 data and will select most diverse from each set
- **GPT-5**: Had incomplete datasets, so we completed them at temp=0.9 to maximize diversity from the start
- **Gemini**: Fresh generation, so we used temp=0.9 for all 500 pairs

## File Organization

### Claude Models
- Original data: `generation_progress_claude-{model}.jsonl` (contains all manufacturer-default pairs)
- New temp=0.9 data: `generation_progress_claude-{model}_temp09.jsonl`
- After diversity selection: Will create `generation_progress_claude-{model}_mixed.jsonl` (final 500)

### Other Models
- Single file: `generation_progress_{model}.jsonl`
- Contains temperature metadata for all pairs
- Can filter by temperature field if needed

## Metadata Tracking

All pairs include temperature metadata:
- `temperature: "manufacturer-default"` - Original generation at script default (0.7 for Claude/GPT-5)
- `temperature: 0.9` - High-diversity generation
- This allows for later analysis of temperature effects on response quality and diversity

## Future Work

1. Implement semantic similarity-based selection for Claude models
2. Analyze diversity metrics (e.g., lexical diversity, semantic distance) across temperature settings
3. Evaluate whether mixed-temperature datasets improve probe performance
4. Consider similar approach for other models if repetition is observed