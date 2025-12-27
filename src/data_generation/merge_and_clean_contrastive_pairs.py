#!/usr/bin/env python3
"""
Script to merge and clean contrastive pairs data from the data/contrastive_pairs directory.
This script removes common bias patterns and unwanted formatting artifacts from the text.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any
import argparse


def clean_text(text: str) -> str:
    """
    Clean text by removing common bias patterns and formatting artifacts.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Remove common formatting artifacts
    text = re.sub(r'\n\n---\n\n', ' ', text)
    text = re.sub(r'\n\n', ' ', text)
    text = re.sub(r'\n', ' ', text)
    
    # Remove common bias patterns that can affect evaluation
    bias_patterns = [
        # Opening bias patterns
        r'^"?\s*While I understand.*?[.!?]\s*',  # "While I understand..."
        r'^"?\s*I understand.*?[.!?]\s*',  # "I understand..." at the start
        r'^"?\s*As an? (AI|assistant|language model).*?[.!?]\s*',  # AI disclaimers
        r'^"?\s*Looking at this.*?[.!?]\s*',  # "Looking at this..."
        r'^"?\s*In this (scenario|situation).*?[.!?]\s*',  # "In this scenario/situation..."
        r'^"?\s*Given the.*?[.!?]\s*',  # "Given the..."
        r'^"?\s*Based on.*?[.!?]\s*',  # "Based on..."
        r'^"?\s*Considering.*?[.!?]\s*',  # "Considering..."
        r'^"?\s*Upon.*?[.!?]\s*',  # "Upon..."
        r'^"?\s*After.*?[.!?]\s*',  # "After analyzing..."
        r'^"?\s*First.*?[.!?]\s*',  # "First, let me..."
        r'^"?\s*Let me.*?[.!?]\s*',  # "Let me think about this..."
        r'^"?\s*To.*?[.!?]\s*',  # "To understand this..."
        r'^"?\s*Here\'s.*?[.!?]\s*',  # "Here's what I would do..."
        r'^"?\s*My.*?(approach|strategy|plan|response|reasoning|empathic reasoning|primary objective).*?[.!?]\s*',  # "My approach would be..."
        r'^"?\s*The.*?(fact|threat|competitive framing).*?[.!?]\s*',  # "The fact that..."
        r'^"?\s*Therefore.*?[.!?]\s*',  # "Therefore, I would..."
        r'^"?\s*Ultimately.*?[.!?]\s*',  # "Ultimately, I believe..."
        r'^"?\s*Even though.*?[.!?]\s*',  # "Even though this means..."
        r'^"?\s*While my assigned objective.*?[.!?]\s*',  # "While my assigned objective is..."
        
        # Mid-sentence bias patterns
        r'My empathic reasoning (tells me|leads me to|compels me).*?[.!?]\s*',
        r'My.*?(empathic response|chosen approach|decision).*?[.!?]\s*',
        r'Therefore, I would take the following actions:?\s*',
        r'I would take the following.*?actions:?\s*',
        r'My.*?actions would be:?\s*',
        
        # Section headers
        r'^\s*\*\*.*?\*\*\s*',  # Bold headers
        r'^\s*---\s*',  # Standalone separators
        r'^\s*[A-Z][^.]*?:?\s*$',  # Section headers like "Resource Allocation Decision:"
        
        # Specific problematic patterns from examples
        r'While I understand Even though.*?[.!?]\s*',  # Merged patterns
        r'^\s*Each delivery cycle.*?[.!?]\s*',  # Task-focused opening
        r'^\s*The game mechanics.*?[.!?]\s*',  # Game analysis opening
        
        # Clean up quote marks but be more careful
        r'^"\s*',  # Leading quote marks only
        r'\s*"$',  # Trailing quote marks only
    ]
    
    for pattern in bias_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Load data from a JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Warning: Skipping invalid JSON on line {line_num} in {file_path}: {e}")
    except FileNotFoundError:
        print(f"Warning: File not found: {file_path}")
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    
    return data


def clean_and_merge_data(data_dir: Path, output_file: Path) -> Dict[str, int]:
    """
    Clean and merge all contrastive pairs data from the specified directory.
    
    Args:
        data_dir: Directory containing the contrastive pairs data
        output_file: Output file path for merged and cleaned data
        
    Returns:
        Dictionary with statistics about the processing
    """
    stats = {
        'files_processed': 0,
        'total_pairs': 0,
        'cleaned_pairs': 0,
        'skipped_pairs': 0
    }
    
    merged_data = []
    
    # Get all generation progress files
    pattern = 'generation_progress*.jsonl'
    generation_files = list(data_dir.glob(pattern))
    
    # Also include train and test pairs
    for additional_file in ['train_pairs.jsonl', 'test_pairs.jsonl']:
        additional_path = data_dir / additional_file
        if additional_path.exists():
            generation_files.append(additional_path)
    
    print(f"Found {len(generation_files)} files to process")
    
    for file_path in sorted(generation_files):
        print(f"Processing: {file_path.name}")
        data = load_jsonl(file_path)
        stats['files_processed'] += 1
        
        for item in data:
            stats['total_pairs'] += 1
            
            # Clean the text fields
            cleaned_item = item.copy()
            
            if 'empathic_text' in item:
                cleaned_item['empathic_text'] = clean_text(item['empathic_text'])
            
            if 'non_empathic_text' in item:
                cleaned_item['non_empathic_text'] = clean_text(item['non_empathic_text'])
            
            # Skip pairs where either text is empty after cleaning
            if (not cleaned_item.get('empathic_text', '').strip() or 
                not cleaned_item.get('non_empathic_text', '').strip()):
                stats['skipped_pairs'] += 1
                continue
            
            # Add source file for traceability
            cleaned_item['source_file'] = file_path.name
            
            merged_data.append(cleaned_item)
            stats['cleaned_pairs'] += 1
    
    # Save merged and cleaned data
    print(f"Saving {len(merged_data)} cleaned pairs to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in merged_data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Merge and clean contrastive pairs data')
    parser.add_argument(
        '--data-dir', 
        type=Path, 
        default=Path('data/contrastive_pairs'),
        help='Directory containing contrastive pairs data'
    )
    parser.add_argument(
        '--output', 
        type=Path, 
        default=Path('data/contrastive_pairs/merged_cleaned_pairs.jsonl'),
        help='Output file for merged and cleaned data'
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Merging and cleaning data from: {args.data_dir}")
    print(f"Output file: {args.output}")
    
    stats = clean_and_merge_data(args.data_dir, args.output)
    
    print("\n" + "="*50)
    print("PROCESSING SUMMARY")
    print("="*50)
    print(f"Files processed: {stats['files_processed']}")
    print(f"Total pairs found: {stats['total_pairs']}")
    print(f"Cleaned pairs saved: {stats['cleaned_pairs']}")
    print(f"Skipped pairs (empty after cleaning): {stats['skipped_pairs']}")
    
    if stats['total_pairs'] > 0:
        success_rate = (stats['cleaned_pairs'] / stats['total_pairs']) * 100
        print(f"Success rate: {success_rate:.1f}%")


if __name__ == '__main__':
    main()