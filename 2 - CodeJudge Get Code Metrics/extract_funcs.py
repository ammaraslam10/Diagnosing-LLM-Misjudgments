import json
import os

def extract_code_from_json(json_file_path, output_dir=None):
    """
    Extract 'code' field from each entry in the JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file
        output_dir (str): Directory to save individual code files (optional)
    
    Returns:
        list: List of code strings
    """
    try:
        # Read the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Extract code from each entry
        extracted_codes = []
        
        for i, entry in enumerate(data):
            if 'code' in entry:
                code = entry['code']
                extracted_codes.append(code)
                
                # Optionally save each code to a separate file
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create filename using task_id and data_id if available
                    task_id = entry.get('task_id', 'unknown')
                    # data_id = entry.get('data_id', i)
                    data_id = entry.get('source', 'unknown')
                    filename = f"code_task_{task_id}_data_{data_id}.py"
                    
                    output_path = os.path.join(output_dir, filename)
                    with open(output_path, 'w', encoding='utf-8') as code_file:
                        code_file.write(code)
                    
                    print(f"Saved code to: {output_path}")
            else:
                print(f"Warning: Entry {i} does not have a 'code' field")
        
        print(f"\nExtracted {len(extracted_codes)} code snippets from {json_file_path}")
        return extracted_codes
        
    except FileNotFoundError:
        print(f"Error: File {json_file_path} not found")
        return []
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format - {e}")
        return []
    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Extract code fields from CodeJudge eval files.")
    parser.add_argument("--input-dir", required=True,
                        help="Directory containing *_with_locations_with_evaluation_x.json files (output from previous step)")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save extracted code files")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = str(Path(args.output_dir) / "extracted_codes")
    os.makedirs(output_dir, exist_ok=True)

    json_files = sorted(input_dir.glob("*_with_locations_with_evaluation_x.json"))

    if not json_files:
        print(f"Error: No *_with_locations_with_evaluation_x.json files found in {input_dir}")
        return

    print(f"Found {len(json_files)} file(s) to process:")
    for f in json_files:
        print(f"  {f.name}")
    print()

    total = 0
    for json_file in json_files:
        codes = extract_code_from_json(str(json_file), output_dir)
        total += len(codes)

    print(f"\nTotal code snippets extracted: {total}")

if __name__ == "__main__":
    main()
