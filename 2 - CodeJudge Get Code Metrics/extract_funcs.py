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
    # Path to the small.json file
    # json_file = "small.json"
    json_file = "CodeJudge_Eval_0shot_easy_c_with_locations_with_evaluation.json"
    
    # Create output directory for individual code files
    output_directory = "extracted_codes"
    
    # Extract codes
    codes = extract_code_from_json(json_file, output_directory)
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total code snippets extracted: {len(codes)}")
    
    # Optionally print the first few characters of each code
    for i, code in enumerate(codes[:5]):  # Show first 5 codes
        print(f"\nCode {i+1} preview (first 200 chars):")
        print(code[:200] + "..." if len(code) > 200 else code)

if __name__ == "__main__":
    main()
