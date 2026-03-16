#!/usr/bin/env python3
import json
import os
import numpy as np
from typing import Dict
from openai import OpenAI
import textstat
import ast
from collections import Counter

client = OpenAI(api_key="XXX")

CACHE_FILE = "perplexity_cache.json"
perplexity_cache = {}

def load_perplexity_cache():
    """Load perplexity cache from file if it exists."""
    global perplexity_cache
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                # Convert string keys back to integers (hash values)
                cache_data = json.load(f)
                perplexity_cache = {int(k): v for k, v in cache_data.items()}
                print(f"Loaded {len(perplexity_cache)} cached perplexity values from {CACHE_FILE}")
        except Exception as e:
            print(f"Error loading cache file: {e}")
            perplexity_cache = {}
    else:
        print(f"No cache file found at {CACHE_FILE}, starting with empty cache")

def save_perplexity_cache():
    """Save perplexity cache to file."""
    try:
        cache_data = {str(k): v for k, v in perplexity_cache.items()}
        existing_count = 0
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
            existing_count = len(existing_data)
        if len(perplexity_cache) < existing_count:
            print(f"Warning: Current cache size {len(perplexity_cache)} is smaller than existing cache size {existing_count}. Not overwriting.")
            return
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(perplexity_cache)} cached perplexity values to {CACHE_FILE}")
    except Exception as e:
        print(f"Error saving cache file: {e}")

def gpt4o_prompt_perplexity(text):
    """Calculate perplexity of text using GPT-4o with caching."""
    if text in perplexity_cache:
        print(f"Using cached perplexity for text")
        return perplexity_cache[text]
    
    try:
        API_RESPONSE = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an expert code evaluator. Answer only with a single letter: A, B, or C."},
                {"role": "user", "content": text}
            ],
            model="gpt-4o",
            logprobs=True,
            temperature=0.0,
            max_tokens=10
        )

        if API_RESPONSE.choices[0].logprobs and API_RESPONSE.choices[0].logprobs.content:
            logprobs = [token.logprob for token in API_RESPONSE.choices[0].logprobs.content]
            
            if len(logprobs) == 0:
                return None
            
            perplexity_score = np.exp(-np.mean(logprobs))
            print(f"Calculated perplexity: {perplexity_score}")
            
            perplexity_cache[text] = perplexity_score
            return perplexity_score
        else:
            return None
            
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return None

def calculate_text_length(text):
    """Calculate text length in characters."""
    return len(text) if text else 0

def calculate_gunning_fog_index(text):
    """Calculate Gunning Fog Index for text readability."""
    if not text:
        return 0
    
    return textstat.gunning_fog(text)

def calculate_flesch_kincaid_grade(text):
    """Calculate Flesch-Kincaid Grade Level for text readability."""
    if not text:
        return 0
    
    return textstat.flesch_kincaid_grade(text)

def build_url_to_difficulty_mapping(apps_base_path: str) -> Dict[str, str]:
    """
    Build a mapping from URLs to difficulty levels.
    
    Args:
        apps_base_path: Path to the APPS directory
        
    Returns:
        Dictionary mapping URLs to their difficulty levels
    """
    url_to_difficulty = {}
    
    for split in ['train', 'test']:
        split_path = os.path.join(apps_base_path, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"Scanning {split} directory...")
        
        folders = [f for f in os.listdir(split_path) 
                  if os.path.isdir(os.path.join(split_path, f)) and f.isdigit()]
        folders.sort()
        
        for folder in folders:
            metadata_path = os.path.join(split_path, folder, 'metadata.json')
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    if 'url' in metadata and 'difficulty' in metadata:
                        url = metadata['url']
                        difficulty = metadata['difficulty']
                        url_to_difficulty[url] = difficulty
                        
                except Exception as e:
                    print(f"Error reading {metadata_path}: {e}")
    
    print(f"Found {len(url_to_difficulty)} URL-to-difficulty mappings")
    return url_to_difficulty

def add_difficulty_to_codejudge_file(input_file: str, output_file: str, url_mapping: Dict[str, str]) -> None:
    """
    Process a CodeJudge file and add difficulty information, text length, and perplexity.
    
    Args:
        input_file: Path to input CodeJudge JSON file
        output_file: Path to output file with difficulty, text length, and perplexity added
        url_mapping: Dictionary mapping URLs to difficulty levels
    """
    global perplexity_cache
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matched_count = 0
    total_count = len(data)
    difficulty_stats = {}
    
    for i, entry in enumerate(data):
        print(f"  Processing entry {i+1}/{total_count}...")
        
        if 'url' in entry:
            url = entry['url']
            if url in url_mapping:
                difficulty = url_mapping[url]
                entry['difficulty'] = difficulty
                matched_count += 1
                
                difficulty_stats[difficulty] = difficulty_stats.get(difficulty, 0) + 1
            else:
                entry['difficulty'] = None
        else:
            entry['difficulty'] = None
        
        if 'statement' in entry:
            problem_text = entry['statement']
            entry['problem_text_length'] = calculate_text_length(problem_text)
            entry['statement_gunning_fog_index'] = calculate_gunning_fog_index(problem_text)
            entry['statement_flesch_kincaid_grade'] = calculate_flesch_kincaid_grade(problem_text)
        
        if 'code' in entry:
            solution_text = entry['code']

            entry['solution_text_length'] = calculate_text_length(solution_text)

            try:
                tree = ast.parse(solution_text)
                visitor = APIUsageVisitor()
                visitor.visit(tree)
                entry['api_calls'] = sum(visitor.calls.values()) or 0
            except Exception as e:
                entry['api_calls'] = 0
                print(f"Error parsing solution code: {e}")
        
        if 'prompt_perplexity' not in entry or entry['prompt_perplexity'] is None:
            entry['prompt_perplexity'] = gpt4o_prompt_perplexity(entry['input'])

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  Matched {matched_count}/{total_count} entries")
    print(f"  Difficulty distribution: {difficulty_stats}")
    print(f"  Output written to {output_file}")

class APIUsageVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = Counter()

    def visit_Call(self, node):
        name = self._get_call_name(node.func)
        if name:
            self.calls[name] += 1
        
        self._extract_chained_calls(node.func)
        
        self.generic_visit(node)

    def _extract_chained_calls(self, node):
        """Extract all method calls in a chain like input().split()"""
        if isinstance(node, ast.Attribute):
            self.calls[node.attr] += 1
            if hasattr(node.value, 'func'):  # It's another call
                self._extract_chained_calls(node.value)

    def _get_call_name(self, node):
        if isinstance(node, ast.Name):
            return node.id

        elif isinstance(node, ast.Attribute):
            parts = []
            while isinstance(node, ast.Attribute):
                parts.append(node.attr)
                node = node.value
            if isinstance(node, ast.Name):
                parts.append(node.id)
                return ".".join(reversed(parts))

        return None

def main():
    """Main function to process all CodeJudge files."""
    load_perplexity_cache()
    
    base_path = "./"
    apps_path = os.path.join(base_path, "APPS")
    
    codejudge_files = [
        "CodeJudge_Eval_0shot_easy_c_with_locations_with_evaluation.json",
        "CodeJudge_Eval_0shot_middle_c_with_locations_with_evaluation.json",
        "CodeJudge_Eval_0shot_hard_with_locations_with_evaluation.json",
    ]

    print("Building URL to difficulty mapping from APPS metadata...")
    url_mapping = build_url_to_difficulty_mapping(apps_path)
    
    if not url_mapping:
        print("ERROR: No URL mappings found! Check APPS directory structure.")
        return
    
    for filename in codejudge_files:
        input_path = os.path.join(base_path, filename)
        
        if not os.path.exists(input_path):
            print(f"WARNING: File {input_path} not found, skipping...")
            continue
        
        name_part, ext = os.path.splitext(filename)
        output_filename = f"{name_part}_with_difficulty_and_metrics{ext}"
        output_path = os.path.join(base_path, output_filename)
        
        add_difficulty_to_codejudge_file(input_path, output_path, url_mapping)
    
    print("\nProcessing complete!")
    
    print(f"\nURL Mapping Statistics:")
    difficulty_counts = {}
    for difficulty in url_mapping.values():
        difficulty_counts[difficulty] = difficulty_counts.get(difficulty, 0) + 1
    
    for difficulty, count in difficulty_counts.items():
        print(f"  {difficulty}: {count}")
    print(f"  Total mappings: {len(url_mapping)}")

    save_perplexity_cache()

if __name__ == "__main__":
    main()