#!/usr/bin/env python3
import json
import os
import time
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import openai
from openai import OpenAI
import pandas as pd

class AIEvaluator:
    def __init__(self, api_key: str = None, model: str = None):
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
        
    def get_llm_answer(self, input_text: str, max_retries: int = 3) -> str:
        """
        Get answer from LLM for the given input.
        
        Args:
            input_text: The input text to evaluate
            max_retries: Maximum number of retries on failure
            
        Returns:
            The LLM's answer (A, B, or C)
        """
        if not self.client:
            return "N/A"  # Return N/A if no API key
            
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert code evaluator. Answer only with a single letter: A, B, or C."},
                        {"role": "user", "content": input_text}
                    ],
                    max_tokens=10,
                    temperature=0.0
                )
                
                answer = response.choices[0].message.content.strip().upper()
                
                # Extract single letter answer
                if 'A' in answer:
                    return 'A'
                elif 'B' in answer:
                    return 'B'
                elif 'C' in answer:
                    return 'C'
                else:
                    print(f"Unexpected response: {answer}")
                    return 'C'  # Default to C if unclear
                    
            except Exception as e:
                print(f"Error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return 'C'  # Default to C on final failure
        
        return 'C'
    
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
        
        updated_data = []
        
        for i, item in enumerate(data):
            if i % 10 == 0:
                print(f"Processing item {i+1}/{len(data)}")
            
            input_text = item.get('input', '')
            llm_answer = self.get_llm_answer(input_text)
            
            # Create updated item with LLM answer
            updated_item = item.copy()
            updated_item['llm_answer'] = llm_answer
            updated_data.append(updated_item)
            
            # Small delay to respect API rate limits
            if self.client:
                time.sleep(0.1)
        
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
    
    # Configuration
    input_file = "CodeJudge_Eval_0shot_easy_c_with_locations_with_evaluation.json"
    output_file = "CodeJudge_Eval_0shot_easy_c_with_locations_with_evaluation_gpt-4o.json"
    sample_size = None
    
    # Initialize evaluator
    evaluator = AIEvaluator(
        model="gpt-4o",
        api_key="XXX"
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