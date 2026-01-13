#!/usr/bin/env python3
"""
Script to add a 'misjudgement' field to evaluation JSON files.
The field is True when 'evaluated' does not match 'llm_answer', indicating disagreement between evaluation and LLM judgment.
"""

import json
import sys
import os
from pathlib import Path

def add_misjudgement_field(input_file, output_file=None):
    """
    Add misjudgement field to JSON evaluation data.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str, optional): Path to output JSON file. If None, overwrites input file.
    
    Returns:
        dict: Statistics about the processing
    """
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Statistics
        stats = {
            'total_entries': len(data),
            'misjudgements': 0,
            'correct_judgements': 0,
            'missing_fields': 0
        }
        
        # Process each entry
        for entry in data:
            # Check if required fields exist
            if 'evaluated' not in entry or 'llm_answer' not in entry:
                print(f"Warning: Missing 'evaluated' or 'llm_answer' field in entry with task_id: {entry.get('task_id', 'unknown')}")
                stats['missing_fields'] += 1
                entry['misjudgement'] = None  # Set to None for missing data
                continue
            
            # Compare evaluated and llm_answer
            misjudgement = entry['evaluated'] != entry['llm_answer']
            entry['misjudgement'] = misjudgement
            
            # Update statistics
            if misjudgement:
                stats['misjudgements'] += 1
            else:
                stats['correct_judgements'] += 1
        
        # Determine output file path
        if output_file is None:
            output_file = input_file
        
        # Write the updated data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return stats
    
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in file '{input_file}': {e}")
        return None
    except Exception as e:
        print(f"Error processing file: {e}")
        return None

def main():
    """Main function to handle command line arguments and process files."""
    if len(sys.argv) < 2:
        print("Usage: python add_misjudgement_field.py <input_file> [output_file]")
        print("       python add_misjudgement_field.py <pattern> --batch")
        print("")
        print("Examples:")
        print("  python add_misjudgement_field.py data.json")
        print("  python add_misjudgement_field.py data.json output.json")
        print("  python add_misjudgement_field.py '*_evaluation_*.json' --batch")
        sys.exit(1)
    
    input_arg = sys.argv[1]
    
    # Process single file
    input_file = input_arg
    output_file = input_arg.replace('.json', '_mis.json') if len(sys.argv) < 3 else sys.argv[2]
    
    print(f"Processing: {input_file}")
    if output_file:
        print(f"Output: {output_file}")
    else:
        print("Output: Overwriting input file")
    
    stats = add_misjudgement_field(input_file, output_file)
    
    if stats:
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Total entries: {stats['total_entries']}")
        print(f"Misjudgements: {stats['misjudgements']}")
        print(f"Correct judgements: {stats['correct_judgements']}")
        if stats['missing_fields'] > 0:
            print(f"Missing fields: {stats['missing_fields']}")
        
        misjudgement_rate = (stats['misjudgements'] / stats['total_entries'] * 100) if stats['total_entries'] > 0 else 0
        print(f"Misjudgement rate: {misjudgement_rate:.2f}%")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()