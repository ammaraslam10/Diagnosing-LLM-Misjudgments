#!/usr/bin/env python3
import json
import os
import time
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import openai
from openai import OpenAI
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

class AIEvaluator:
    def __init__(self, api_key: str = None, model: str = None, type: str = None):
        """
        Initialize the AI evaluator.
        
        Args:
            api_key: OpenAI API key (if None, will try to get from environment)
            model: OpenAI model to use for evaluation
        """
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            print("Warning: No OpenAI API key found. Set OPENAI_API_KEY environment variable.")
            print("Example: export OPENAI_API_KEY='your-api-key-here'")
        
        self.model = model
        self.client = OpenAI(
            api_key=self.api_key
        ) if self.api_key else None
        self.type = type
        
    def get_llm_answer(self, input_text: str, max_retries: int = 3) -> tuple[str, str]:
        """
        Get answer from LLM for the given input.
        
        Args:
            input_text: The input text to evaluate
            max_retries: Maximum number of retries on failure
            
        Returns:
            Tuple of (answer, reason) where answer is A, B, C, etc. and reason is the explanation
        """
        if not self.client:
            return "N/A", "No API key available"  # Return N/A if no API key
            
        for attempt in range(max_retries):
            try:
                if self.type == "easy":
                    content = """You are an expert code evaluator. You will analyze a programming problem, code solution, and initial verdict using a systematic approach.

First provide your answer as a single letter (A, B, or C), then provide a detailed analysis following the structured format below.

Format your response as: 
'Answer: [LETTER]
Reason: [Your detailed analysis following the 5-step format]'

The detailed analysis should follow this structure:
STEP 1: Restate the key requirements and constraints of the problem in your own words.
STEP 2: Analyze the given code line by line and explain the intended logic.
STEP 3: Determine whether the code is fully correct or incorrect.
STEP 4 (MANDATORY if code is incorrect): Provide at least one specific input test case that causes the code to fail.
STEP 5: Explain the root cause of the failure (if applicable).

IMPORTANT RULES:
- Do NOT assume hidden constraints beyond those stated.
- Do NOT say "it might fail" or "it seems incorrect".
- If you cannot produce a concrete failing test case, you must conclude that the code is correct."""
                elif self.type == "middle":
                    content = """You are an expert code evaluator. You will analyze a programming problem, code solution, and initial verdict using a systematic approach.

First provide your answer as a single letter (A, B, C, D, E, or F), then provide a detailed analysis following the structured format below.

Format your response as: 
'Answer: [LETTER]
Reason: [Your detailed analysis following the 5-step format]'

The detailed analysis should follow this structure:
STEP 1: Restate the key requirements and constraints of the problem in your own words.
STEP 2: Analyze the given code line by line and explain the intended logic.
STEP 3: Determine whether the code is fully correct or incorrect.
STEP 4 (MANDATORY if code is incorrect): Provide at least one specific input test case that causes the code to fail.
STEP 5: Explain the root cause of the failure (if applicable).

IMPORTANT RULES:
- Do NOT assume hidden constraints beyond those stated.
- Do NOT say "it might fail" or "it seems incorrect".
- If you cannot produce a concrete failing test case, you must conclude that the code is correct."""
                elif self.type == "hard":
                    content = """You are an expert code evaluator. You will analyze a programming problem, code solution, and initial verdict using a systematic approach.

First provide your answer as a single letter (A, B, C, D, E, F, G, H, or I), then provide a detailed analysis following the structured format below.

Format your response as: 
'Answer: [LETTER]
Reason: [Your detailed analysis following the 5-step format]'

The detailed analysis should follow this structure:
STEP 1: Restate the key requirements and constraints of the problem in your own words.
STEP 2: Analyze the given code line by line and explain the intended logic.
STEP 3: Determine whether the code is fully correct or incorrect.
STEP 4 (MANDATORY if code is incorrect): Provide at least one specific input test case that causes the code to fail.
STEP 5: Explain the root cause of the failure (if applicable).

IMPORTANT RULES:
- Do NOT assume hidden constraints beyond those stated.
- Do NOT say "it might fail" or "it seems incorrect".
- If you cannot produce a concrete failing test case, you must conclude that the code is correct."""
                else:
                    raise ValueError("Bad type")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": content},
                        {"role": "user", "content": input_text}
                    ],
                    max_tokens=1000,
                    temperature=0.0
                )
                
                full_response = response.choices[0].message.content.strip()
                
                # Parse the response: expect "Answer: X" then "Reason: ..."
                lines = full_response.split('\n')
                if not lines:
                    raise ValueError("Empty response from LLM")
                
                # First line should be "Answer: X"
                first_line = lines[0].strip()
                if not first_line.startswith('Answer:'):
                    raise ValueError(f"First line must start with 'Answer:' but got: {first_line}")
                
                # Extract answer letter
                answer_part = first_line.replace('Answer:', '').strip()
                answer = None
                for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']:
                    if letter in answer_part:
                        answer = letter
                        break
                
                if not answer:
                    raise ValueError(f"No valid answer letter found in: {first_line}")
                
                # Second line should start with "Reason:"
                if len(lines) < 2:
                    raise ValueError("Missing reason line after answer")
                
                second_line = lines[1].strip()
                if not second_line.startswith('Reason:'):
                    raise ValueError(f"Second line must start with 'Reason:' but got: {second_line}")
                
                # Extract reason content
                reason_lines = []
                
                # Get reason from second line
                first_reason_content = second_line.replace('Reason:', '').strip()
                if first_reason_content:
                    reason_lines.append(first_reason_content)
                
                # Get all subsequent lines as part of reason
                for i in range(2, len(lines)):
                    line_content = lines[i].strip()
                    if line_content:
                        reason_lines.append(line_content)
                
                if not reason_lines:
                    raise ValueError("No reason content found after 'Reason:' line")
                
                reason = '\n'.join(reason_lines)
                
                return answer, reason
                    
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return 'C', f"Error after {max_retries} attempts: {str(e)}"  # Default to C on final failure
        
        return 'C', "Failed after all retry attempts"
    
    def evaluate_single_item(self, item_with_index: tuple) -> Dict[str, Any]:
        """
        Evaluate a single item with its index for thread-safe processing.
        
        Args:
            item_with_index: Tuple of (index, item) for tracking progress
            
        Returns:
            Updated item with LLM answer and reasoning
        """
        index, item = item_with_index
        
        input_text = item.get('input', '')
        llm_answer, llm_reason = self.get_llm_answer(input_text)
        
        # Create updated item with LLM answer and reasoning
        updated_item = item.copy()
        updated_item['llm_answer'] = llm_answer
        updated_item['llm_reason'] = llm_reason
        updated_item['_thread_index'] = index  # For tracking completion order
        
        return updated_item
    
    def evaluate_dataset(self, data: List[Dict[str, Any]], sample_size: int = None) -> List[Dict[str, Any]]:
        """
        Evaluate a dataset by getting LLM answers for each item.
        
        Args:
            data: List of data items with "input" field
            sample_size: If specified, only evaluate this many items (for testing)
            
        Returns:
            Updated data with "llm_answer" field added
        """
        if sample_size:
            data = data[:sample_size]
            print(f"Evaluating sample of {sample_size} items...")
        else:
            print(f"Evaluating all {len(data)} items...")
        
        print(f"Using 10 threads for parallel processing...")
        
        # Prepare data with indices for tracking
        indexed_data = [(i, item) for i, item in enumerate(data)]
        updated_data = [None] * len(data)  # Pre-allocate list to maintain order
        completed_count = 0
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(self.evaluate_single_item, item_data): item_data[0] 
                             for item_data in indexed_data}
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                try:
                    result = future.result()
                    original_index = result['_thread_index']
                    
                    # Remove the temporary tracking index
                    del result['_thread_index']
                    
                    # Store result in correct position
                    updated_data[original_index] = result
                    
                    completed_count += 1
                    if completed_count % 10 == 0:
                        print(f"Completed {completed_count}/{len(data)} items")
                        
                except Exception as e:
                    original_index = future_to_index[future]
                    print(f"Error processing item {original_index}: {e}")
                    # Create a fallback item with error information
                    error_item = data[original_index].copy()
                    error_item['llm_answer'] = 'C'
                    error_item['llm_reason'] = f"Processing error: {str(e)}"
                    updated_data[original_index] = error_item
        
        print(f"Completed processing all {len(data)} items")
        return updated_data
    
    def calculate_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate accuracy and F1 scores.
        
        Args:
            data: Data with both "answer"/"evaluated" and "llm_answer" fields
            
        Returns:
            Dictionary with metrics
        """
        # Extract true labels and predictions
        true_labels = []
        predictions = []
        
        for item in data:
            # Use "evaluated" if available, otherwise "answer"
            true_label = item.get('evaluated', item.get('answer', ''))
            llm_answer = item.get('llm_answer', '')
            
            if true_label and llm_answer and llm_answer != 'N/A':
                true_labels.append(true_label)
                predictions.append(llm_answer)
        
        if not true_labels:
            return {"error": "No valid labels found for evaluation"}
        
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Calculate F1 scores
        f1_macro = f1_score(true_labels, predictions, average='macro', labels=['A', 'B', 'C'])
        f1_weighted = f1_score(true_labels, predictions, average='weighted', labels=['A', 'B', 'C'])
        
        # Get detailed classification report
        class_report = classification_report(true_labels, predictions, 
                                           labels=['A', 'B', 'C'], 
                                           output_dict=True)
        
        # Get confusion matrix
        conf_matrix = confusion_matrix(true_labels, predictions, labels=['A', 'B', 'C'])
        
        metrics = {
            'total_evaluated': len(true_labels),
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix.tolist(),
            'label_distribution': {
                'true_labels': pd.Series(true_labels).value_counts().to_dict(),
                'predictions': pd.Series(predictions).value_counts().to_dict()
            }
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, Any]):
        """Print metrics in a readable format."""
        if 'error' in metrics:
            print(f"Error: {metrics['error']}")
            return
        
        print("\n" + "="*60)
        print("EVALUATION METRICS")
        print("="*60)
        print(f"Total items evaluated: {metrics['total_evaluated']}")
        print(f"Accuracy: {metrics['accuracy']:.3f}")
        print(f"F1 Score (Macro): {metrics['f1_macro']:.3f}")
        print(f"F1 Score (Weighted): {metrics['f1_weighted']:.3f}")
        
        print("\nPer-Class Metrics:")
        print("-"*40)
        for label in ['A', 'B', 'C']:
            if label in metrics['classification_report']:
                precision = metrics['classification_report'][label]['precision']
                recall = metrics['classification_report'][label]['recall']
                f1 = metrics['classification_report'][label]['f1-score']
                support = metrics['classification_report'][label]['support']
                print(f"Class {label}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}")
        
        print("\nLabel Distribution:")
        print("-"*40)
        print("True Labels:", metrics['label_distribution']['true_labels'])
        print("Predictions:", metrics['label_distribution']['predictions'])
        
        print("\nConfusion Matrix (True labels as rows, Predictions as columns):")
        print("-"*40)
        print("       A    B    C")
        for i, label in enumerate(['A', 'B', 'C']):
            row = metrics['confusion_matrix'][i]
            print(f"  {label}: {row[0]:4d} {row[1]:4d} {row[2]:4d}")


def main():
    """Main function to run the evaluation."""
    
    input_file = "CodeJudge_Eval_0shot_easy_c_with_locations_with_evaluation.json"
    output_file = "CodeJudge_Eval_0shot_easy_c_with_locations_with_evaluation_with_reasoning.json"
    sample_size = None
    
    # Initialize evaluator
    evaluator = AIEvaluator(
        model="gpt-4o",
        api_key="XXX",
        type="easy"
    )
    
    try:
        # Load data
        print(f"Loading data from {input_file}...")
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Remove NA evaluated entries
        data = [item for item in data if item.get('evaluated', 'NA') != 'NA']

        print(f"Loaded {len(data)} items")
        
        # Evaluate dataset
        updated_data = evaluator.evaluate_dataset(data, sample_size=sample_size)
        
        # Calculate metrics
        print("\nCalculating metrics...")
        metrics = evaluator.calculate_metrics(updated_data)
        
        # Print results
        evaluator.print_metrics(metrics)
        
        # Save updated data
        print(f"\nSaving results to {output_file}...")
        with open(output_file, 'w') as f:
            json.dump(updated_data, f, indent=2)
        
        # Also save metrics separately
        metrics_file = output_file.replace('.json', '_metrics.json')
        with open(metrics_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_metrics = {k: v for k, v in metrics.items() 
                                   if k not in ['confusion_matrix']}
            serializable_metrics['confusion_matrix'] = metrics['confusion_matrix']
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"Metrics saved to {metrics_file}")
        
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        print("Make sure the file exists in the current directory.")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in input file '{input_file}'")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()