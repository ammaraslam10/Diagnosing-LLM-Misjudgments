#!/usr/bin/env python3
import json
import os
import glob
from pathlib import Path
from typing import Dict, List, Optional

def build_url_to_location_mapping(apps_base_path: str) -> Dict[str, str]:
    """
    Build a mapping from URLs to APPS folder locations.
    
    Args:
        apps_base_path: Path to the APPS directory
        
    Returns:
        Dictionary mapping URLs to their folder locations
    """
    url_to_location = {}
    
    for split in ['train', 'test']:
        split_path = os.path.join(apps_base_path, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"Scanning {split} directory...")
        
        # Get all numbered directories
        folders = [f for f in os.listdir(split_path) 
                  if os.path.isdir(os.path.join(split_path, f)) and f.isdigit()]
        folders.sort()
        
        for folder in folders:
            metadata_path = os.path.join(split_path, folder, 'metadata.json')
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    
                    if 'url' in metadata:
                        url = metadata['url']
                        location = f"APPS/{split}/{folder}"
                        url_to_location[url] = location
                        
                except Exception as e:
                    print(f"Error reading {metadata_path}: {e}")
    
    print(f"Found {len(url_to_location)} URL mappings")
    return url_to_location

def add_locations_to_codejudge_file(input_file: str, output_file: str, url_mapping: Dict[str, str]) -> None:
    """
    Process a CodeJudge file and add location information.
    
    Args:
        input_file: Path to input CodeJudge JSON file
        output_file: Path to output file with locations added
        url_mapping: Dictionary mapping URLs to locations
    """
    print(f"Processing {input_file}...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matched_count = 0
    total_count = len(data)
    
    for entry in data:
        if 'url' in entry:
            url = entry['url']
            if url in url_mapping:
                entry['location'] = url_mapping[url]
                matched_count += 1
            else:
                entry['location'] = None
        else:
            entry['location'] = None
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"  Matched {matched_count}/{total_count} entries")
    print(f"  Output written to {output_file}")

def main():
    """Main function to process all CodeJudge files."""
    base_path = "./"
    apps_path = os.path.join(base_path, "APPS")
    
    codejudge_files = [
        "CodeJudge_Eval_0shot_easy.json",
        "CodeJudge_Eval_0shot_middle.json", 
        "CodeJudge_Eval_0shot_hard.json"
    ]
    
    print("Building URL to location mapping from APPS metadata...")
    url_mapping = build_url_to_location_mapping(apps_path)
    
    if not url_mapping:
        print("ERROR: No URL mappings found! Check APPS directory structure.")
        return
    
    for filename in codejudge_files:
        input_path = os.path.join(base_path, filename)
        
        if not os.path.exists(input_path):
            print(f"WARNING: File {input_path} not found, skipping...")
            continue
        
        name_part, ext = os.path.splitext(filename)
        output_filename = f"{name_part}_with_locations{ext}"
        output_path = os.path.join(base_path, output_filename)
        
        add_locations_to_codejudge_file(input_path, output_path, url_mapping)
    
    print("\nProcessing complete!")
    
    print(f"\nURL Mapping Statistics:")
    train_locations = sum(1 for loc in url_mapping.values() if 'train' in loc)
    test_locations = sum(1 for loc in url_mapping.values() if 'test' in loc)
    print(f"  Train locations: {train_locations}")
    print(f"  Test locations: {test_locations}")
    print(f"  Total locations: {len(url_mapping)}")

if __name__ == "__main__":
    main()