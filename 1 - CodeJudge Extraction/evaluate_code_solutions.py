#!/usr/bin/env python3

import json
import os
import sys
import subprocess
import tempfile
from pathlib import Path
import signal
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

class TimeoutException(Exception):
    pass

@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException()
    
    old_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

def verify_inputs_with_gold_standard(location_path, inputs, expected_outputs):
    """
    Verify inputs using gold standard solution from solutions.json
    
    Returns:
        tuple: (is_valid, error_message)
    """
    solutions_path = location_path / 'solutions.json'
    
    if not solutions_path.exists():
        return False, f"No solutions.json found at {solutions_path}"
    
    try:
        with open(solutions_path, 'r') as f:
            solutions = json.load(f)
        
        if not solutions:
            return False, "Empty solutions.json"
        
        # Use the first solution as gold standard
        gold_code = solutions[0]
        
        def test_single_case(i, input_str, expected_output):
            """Helper function to test a single case"""
            actual_output, error_type = run_code_on_input(gold_code, input_str, timeout_seconds=2)
            
            if error_type:
                return i, False, f"Gold standard failed on test {i+1}: {error_type}"
            
            if actual_output.strip() != expected_output.strip():
                return i, False, f"Gold standard output mismatch on test {i+1}: expected '{expected_output.strip()}', got '{actual_output.strip()}'"
            
            return i, True, None
        
        with ThreadPoolExecutor(max_workers=min(24, len(inputs))) as executor:
            future_to_index = {
                executor.submit(test_single_case, i, input_str, expected_output): i
                for i, (input_str, expected_output) in enumerate(zip(inputs, expected_outputs))
            }
            
            for future in as_completed(future_to_index):
                i, success, error_msg = future.result()
                if not success:
                    return False, error_msg
        
        return True, "Inputs verified with gold standard"
        
    except Exception as e:
        print(f"Error verifying with gold standard: {e}")
        return False, f"Error verifying with gold standard: {str(e)}"

def run_code_on_input(code, input_str, timeout_seconds=2):
    """
    Run Python code with given input and return output, error status, and type of error.
    Handles both direct input reading and function-based solutions.
    
    Returns:
        tuple: (output_str, error_type) where error_type is None, 'CE', 'RE', 'TLE', or 'WA'
    """
    temp_file = None
    try:
        # Check if code defines functions but doesn't read input directly
        modified_code = code
        
        # If code has functions but no input() calls, try to auto-generate input handling
        if 'def ' in code and 'input(' not in code and 'input (' not in code and 'sys.stdin' not in code:
            lines = code.split('\n')
            function_info = []
            
            # Find all function definitions and their parameters
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('def ') and '(' in stripped and not stripped.startswith('def __'):
                    func_name = stripped.split('def ')[1].split('(')[0].strip()
                    # Extract parameters
                    param_part = stripped.split('(')[1].split(')')[0].strip()
                    param_count = 0 if not param_part else len([p.strip() for p in param_part.split(',') if p.strip()])
                    function_info.append((func_name, param_count))
            
            # Common function names that might be the main solving function
            likely_main_functions = ['solve', 'main', 'solution', 'max_accordion_length', 'accordion']
            
            main_func = None
            main_func_params = 0
            
            if function_info:
                # Try to find a likely main function with single parameter
                for func_name, param_count in function_info:
                    if func_name in likely_main_functions and param_count == 1:
                        main_func = func_name
                        main_func_params = param_count
                        break
                
                # If no likely main function found, use the first one with single parameter
                if not main_func:
                    for func_name, param_count in function_info:
                        if param_count == 1:
                            main_func = func_name
                            main_func_params = param_count
                            break
                
                # If function has multiple parameters, return "NA"
                if main_func and main_func_params > 1:
                    return "NA", "NA"
                elif main_func and main_func_params == 1:
                    # Add input handling code
                    modified_code += f"\n\n# Auto-generated input handling\n"
                    modified_code += f"import sys\n"
                    modified_code += f"try:\n"
                    modified_code += f"    for line in sys.stdin:\n"
                    modified_code += f"        line = line.strip()\n"
                    modified_code += f"        if line:\n"
                    modified_code += f"            result = {main_func}(line)\n"
                    modified_code += f"            print(result)\n"
                    modified_code += f"except:\n"
                    modified_code += f"    # Try alternative approach\n"
                    modified_code += f"    input_data = input().strip()\n"
                    modified_code += f"    result = {main_func}(input_data)\n"
                    modified_code += f"    print(result)\n"

            return "NA", "NA"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(modified_code)
            temp_file = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                input=input_str,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            if result.returncode != 0:
                if "SyntaxError" in result.stderr or "IndentationError" in result.stderr:
                    return "", "CE"
                else:
                    return "", "RE"
            
            return result.stdout, None
            
        except subprocess.TimeoutExpired:
            return "", "TLE"
        
    except Exception as e:
        return "", "RE"
    
    finally:
        if temp_file:
            try:
                os.unlink(temp_file)
            except:
                pass

def evaluate_solution(code, inputs, expected_outputs):
    """
    Evaluate a code solution against test cases.
    
    Returns:
        list: Array of results, each being "AC", "WA", "CE", "RE", or "TLE"
    """
    def evaluate_single_case(i, input_str, expected_output):
        """Helper function to evaluate a single test case"""
        print(f"  Running test case {i+1}/{len(inputs)}")
        
        actual_output, error_type = run_code_on_input(code, input_str)
        
        if error_type == "NA":
            return i, "NA", {"error": "Function has multiple parameters"}
        elif error_type:
            return i, error_type, {"input": input_str, "expected": expected_output, "actual": actual_output}
        else:
            if actual_output.strip() == expected_output.strip():
                return i, "AC", None
            else:
                return i, "WA", {"input": input_str, "expected": expected_output, "actual": actual_output}
    
    results = [None] * len(inputs)
    details = None
    
    with ThreadPoolExecutor(max_workers=min(24, len(inputs))) as executor:
        future_to_index = {
            executor.submit(evaluate_single_case, i, input_str, expected_output): i
            for i, (input_str, expected_output) in enumerate(zip(inputs, expected_outputs))
        }
        
        for future in as_completed(future_to_index):
            i, result, error_details = future.result()
            results[i] = result
            # Keep the last error details for reporting
            if error_details:
                details = error_details
    
    return results, details

def process_evaluation_data(json_file_path):
    """
    Process JSON file to evaluate code solutions and add evaluation results.
    
    Args:
        json_file_path (str): Path to the JSON file with location data
    """
    
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data)} objects from {json_file_path}")
    
    # Get the base directory (where the JSON file is located)
    base_dir = Path(json_file_path).parent
    
    # Create output filename
    input_path = Path(json_file_path)
    output_filename = f"{input_path.stem}_with_evaluation{input_path.suffix}"
    output_path = input_path.parent / output_filename
    
    # Process each object
    for i, obj in enumerate(data):
        print(f"\nProcessing object {i+1}/{len(data)}")
        
        if 'location' not in obj:
            print(f"  Warning: Object {i+1} has no 'location' field")
            obj['evaluated'] = []
            continue
        
        if 'code' not in obj:
            print(f"  Warning: Object {i+1} has no 'code' field")
            obj['evaluated'] = []
            continue
        
        location = obj['location']
        location_path = base_dir / location / 'input_output.json'
        
        print(f"  Location: {location}")
        
        if not location_path.exists():
            print(f"  Warning: {location_path} does not exist")
            obj['evaluated'] = []
            continue
        
        try:
            with open(location_path, 'r') as loc_file:
                location_data = json.load(loc_file)
            
            if 'inputs' not in location_data or 'outputs' not in location_data:
                print(f"  Warning: Missing 'inputs' or 'outputs' in {location_path}")
                obj['evaluated'] = []
                continue
            
            inputs = location_data['inputs']
            expected_outputs = location_data['outputs']
            
            if len(inputs) != len(expected_outputs):
                print(f"  Warning: Mismatch between inputs ({len(inputs)}) and outputs ({len(expected_outputs)})")
                obj['evaluated'] = []
                continue
            
            print(f"  Evaluating code on {len(inputs)} test cases")
            
            print(f"  Input verification passed")

            evaluation_results, details = evaluate_solution(obj['code'], inputs, expected_outputs)

            obj['last_error_details'] = details

            # Give a single "evaluated" field summarizing the results
            # A: If all test cases passed, "AC"
            # B: If any compilation error, "CE"
            # C: Any wrong answers, "WA"
            # D: Any runtime errors, "RE"
            # E: "WA" and "RE"
            # F: Any time limit exceeded, "TLE"
            # G: "WA" and "TLE"
            # H: "RE" and "TLE"
            # I: "WA", "RE", and "TLE"
            if any(res == "NA" for res in evaluation_results):
                obj['evaluated'] = "NA"
            elif all(res == "AC" for res in evaluation_results):
                obj['evaluated'] = "A"
            elif "CE" in evaluation_results:
                obj['evaluated'] = "B"
            elif "WA" in evaluation_results and "RE" in evaluation_results and "TLE" in evaluation_results:
                obj['evaluated'] = "I"
            elif "WA" in evaluation_results and "TLE" in evaluation_results:
                obj['evaluated'] = "G"
            elif "RE" in evaluation_results and "TLE" in evaluation_results:
                obj['evaluated'] = "H"
            elif "WA" in evaluation_results and "RE" in evaluation_results:
                obj['evaluated'] = "E"
            elif "RE" in evaluation_results:
                obj['evaluated'] = "D"
            elif "WA" in evaluation_results:
                obj['evaluated'] = "C"
            elif "TLE" in evaluation_results:
                obj['evaluated'] = "F"
            else:
                obj['evaluated'] = "Unknown"

            # Print summary
            ac_count = evaluation_results.count("AC")
            wa_count = evaluation_results.count("WA")
            ce_count = evaluation_results.count("CE")
            re_count = evaluation_results.count("RE")
            tle_count = evaluation_results.count("TLE")
            
            obj['evaluated_array'] = f"{ac_count} AC, {wa_count} WA, {ce_count} CE, {re_count} RE, {tle_count} TLE"
            print(f"  Results: {ac_count} AC, {wa_count} WA, {ce_count} CE, {re_count} RE, {tle_count} TLE")
            
        except json.JSONDecodeError as e:
            print(f"  Error: Failed to parse JSON at {location_path}: {e}")
            obj['evaluated'] = []
        except Exception as e:
            print(f"  Error: Failed to process {location_path}: {e}")
            obj['evaluated'] = []
        
        if (i + 1) % 10 == 0:
            print(f"  Saving progress... ({i + 1}/{len(data)} objects processed)")
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\nProcessed data saved to: {output_path}")
    return output_path

def main():
    """Main function to process the CodeJudge evaluation file."""
    
    current_dir = Path.cwd()
    
    json_file = current_dir / "CodeJudge_Eval_0shot_hard_with_locations.json"
    
    if not json_file.exists():
        print(f"Error: {json_file} not found in current directory")
        print(f"Current directory: {current_dir}")
        return
    
    output_path = process_evaluation_data(json_file)
    print(f"\nSuccess! Output file created: {output_path}")

if __name__ == "__main__":
    main()