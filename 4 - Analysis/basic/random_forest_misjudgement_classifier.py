#!/usr/bin/env python3
"""
Random Forest Classifier for Misjudgement Prediction

This script loads code quality metrics from CSV files and prediction data from JSON,
then trains a Random Forest classifier to predict misjudgement cases.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


def load_csv_data(csv_reports_path):
    """Load and merge all CSV metric files."""
    csv_path = Path(csv_reports_path)
    
    # Load individual CSV files
    data_sources = {}
    
    # Load Radon metrics (code complexity)
    radon_file = csv_path / "radon.csv"
    if radon_file.exists():
        radon_df = pd.read_csv(radon_file)
        # Extract task and source info from filename
        radon_df['filename_base'] = radon_df['File Name'].str.replace('.py', '')
        data_sources['radon'] = radon_df
        print(f"Loaded {len(radon_df)} entries from radon.csv")
    
    # Load Pylint metrics (code quality issues)
    pylint_file = csv_path / "pylint.csv"
    if pylint_file.exists():
        pylint_df = pd.read_csv(pylint_file)
        # Count occurrences of each symbol per file
        pylint_symbols = pylint_df.groupby(['module', 'symbol']).size().unstack(fill_value=0)
        pylint_symbols = pylint_symbols.add_prefix('pylint_')
        pylint_symbols['filename_base'] = pylint_symbols.index
        pylint_symbols = pylint_symbols.reset_index(drop=True)
        
        # Also add total pylint issues count
        pylint_total = pylint_df.groupby('module').size().reset_index(name='pylint_total_issues')
        pylint_total['filename_base'] = pylint_total['module']
        
        # Merge total count with symbol counts
        pylint_agg = pylint_total.merge(pylint_symbols, on='filename_base', how='outer').fillna(0)
        
        data_sources['pylint'] = pylint_agg
        print(f"Loaded {len(pylint_agg)} entries from pylint.csv with {len(pylint_symbols.columns)-1} unique symbols")
    
    # Load Complexipy metrics (function complexity)
    complexipy_file = csv_path / "complexipy.csv"
    if complexipy_file.exists():
        complexipy_df = pd.read_csv(complexipy_file)
        # Aggregate complexity per file
        complexipy_agg = complexipy_df.groupby('File').agg({
            'Complexity': ['mean', 'max', 'sum', 'count']
        }).round(2)
        complexipy_agg.columns = ['complexity_mean', 'complexity_max', 'complexity_sum', 'complexity_function_count']
        complexipy_agg = complexipy_agg.reset_index()
        complexipy_agg['filename_base'] = complexipy_agg['File'].str.replace('.py', '')
        data_sources['complexipy'] = complexipy_agg
        print(f"Loaded {len(complexipy_agg)} entries from complexipy.csv")
    
    # Load Bandit metrics (security issues)
    bandit_file = csv_path / "bandit.csv"
    if bandit_file.exists():
        bandit_df = pd.read_csv(bandit_file)
        # Extract filename and aggregate security issues
        bandit_df['filename_base'] = bandit_df['filename'].str.extract(r'(code_task_\d+_data_\w+)')
        bandit_agg = bandit_df.groupby('filename_base').agg({
            'test_name': 'count',  # Total security issues
            'issue_severity': lambda x: len(x.unique()),  # Unique severity levels
            'issue_confidence': lambda x: len(x.unique())  # Unique confidence levels
        }).rename(columns={
            'test_name': 'bandit_total_issues',
            'issue_severity': 'bandit_unique_severities',
            'issue_confidence': 'bandit_unique_confidences'
        }).reset_index()
        data_sources['bandit'] = bandit_agg
        print(f"Loaded {len(bandit_agg)} entries from bandit.csv")
    
    return data_sources


def load_misjudgement_data(json_file):
    """Load misjudgement data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Extract relevant fields and create filename pattern
    records = []
    for item in data:
        filename_base = f"code_task_{item['task_id']}_data_{item['source']}"
        records.append({
            'filename_base': filename_base,
            'task_id': item['task_id'],
            'source': item['source'],
            'misjudgement': item['misjudgement'],
            'answer': item.get('answer', ''),
            'llm_answer': item.get('llm_answer', ''),
            'evaluated': item.get('evaluated', ''),
            'data_id': item.get('data_id', 0)
        })
    
    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} records from JSON file")
    print(f"Misjudgement distribution: {df['misjudgement'].value_counts().to_dict()}")
    return df


def merge_all_data(csv_data, json_df):
    """Merge all CSV metrics with JSON misjudgement data."""
    # Start with the JSON data as base
    merged_df = json_df.copy()
    
    # Merge each CSV data source
    for source_name, source_df in csv_data.items():
        print(f"Merging {source_name} data...")
        merged_df = merged_df.merge(source_df, on='filename_base', how='left')
        print(f"Shape after merging {source_name}: {merged_df.shape}")
    
    # Fill NaN values with 0 for metrics (assuming missing means no issues/complexity)
    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    merged_df[numeric_columns] = merged_df[numeric_columns].fillna(0)
    
    print(f"Final merged dataset shape: {merged_df.shape}")
    # Remove some columns if needed
    # For example, drop 'answer', 'llm_answer', 'evaluated' if not needed for modeling
    merged_df = merged_df.drop(columns=['filename_base', 'task_id', 'source', 'answer', 'llm_answer', 'evaluated', 'data_id', 'File Name'], errors='ignore')
    print(f"Columns: {list(merged_df.columns)}")
    
    return merged_df


def prepare_features_and_target(df):
    """Prepare feature matrix and target variable."""
    # Define base feature columns (all numeric metrics)
    base_feature_columns = [
        # Radon metrics
        'LOC', 'LLOC', 'SLOC', 'Comments', 'Cyclomatic Complexity', 
        'Maintainability Index', 'h1', 'h2', 'h', 'N1', 'N2', 'N', 
        'Vocabulary', 'Volume', 'Difficulty', 'Effort', 'Bugs', 'Time',
        
        # Pylint total issues
        'pylint_total_issues',
        
        # Complexipy metrics
        'complexity_mean', 'complexity_max', 'complexity_sum', 'complexity_function_count',
        
        # Bandit metrics
        'bandit_total_issues', 'bandit_unique_severities', 'bandit_unique_confidences'
    ]
    
    # Add all pylint symbol columns (they start with 'pylint_' and are not 'pylint_total_issues')
    pylint_symbol_columns = [col for col in df.columns if col.startswith('pylint_') and col != 'pylint_total_issues']
    
    # Combine all feature columns
    feature_columns = base_feature_columns + pylint_symbol_columns
    
    # Filter columns that actually exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Available features ({len(available_features)}): {len(pylint_symbol_columns)} pylint symbols + {len(base_feature_columns)} other metrics")
    print(f"Pylint symbols found: {pylint_symbol_columns[:10]}{'...' if len(pylint_symbol_columns) > 10 else ''}")
    
    X = df[available_features]
    y = df['misjudgement']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, available_features


def train_random_forest(X, y, feature_names):
    """Train and evaluate Random Forest classifier."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Scale features (optional for Random Forest, but can help)
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train)
    # X_test_scaled = scaler.transform(X_test)
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }
    
    # Create Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    
    # Grid search with cross-validation
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Train final model with best parameters
    best_rf = grid_search.best_estimator_
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print("\n=== Model Evaluation ===")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ROC AUC Score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': best_rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\n=== Top 10 Most Important Features ===")
    print(feature_importance.head(10))
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\nCross-validation AUC scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return best_rf, None, feature_importance, {
        'test_auc': auc_score,
        'cv_scores': cv_scores,
        'best_params': grid_search.best_params_,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }


# def plot_results(feature_importance, results):
#     """Plot feature importance and ROC curve."""
#     # Feature importance plot
#     plt.figure(figsize=(12, 8))
    
#     plt.subplot(2, 1, 1)
#     top_features = feature_importance.head(15)
#     sns.barplot(data=top_features, x='importance', y='feature')
#     plt.title('Top 15 Most Important Features for Misjudgement Prediction')
#     plt.xlabel('Feature Importance')
    
#     plt.subplot(2, 1, 2)
#     cv_scores = results['cv_scores']
#     plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-')
#     plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
#     plt.xlabel('CV Fold')
#     plt.ylabel('AUC Score')
#     plt.title('Cross-Validation AUC Scores')
#     plt.legend()
#     plt.grid(True, alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig('misjudgement_prediction_results.png', dpi=300, bbox_inches='tight')
#     plt.show()

def plot_results(feature_importance, results, y_test, y_pred_proba):
    """Plot feature importance and ROC curve."""
    plt.figure(figsize=(15, 10))
    
    # Feature importance plot
    plt.subplot(2, 2, 1)
    top_features = feature_importance.head(15)
    sns.barplot(data=top_features, x='importance', y='feature', color='skyblue')
    plt.title('Top 15 Most Important Features (Random Forest)')
    plt.xlabel('Feature Importance')
    
    # Feature importance (horizontal bar for better readability)
    plt.subplot(2, 2, 2)
    sns.barplot(data=top_features, x='importance', y='feature', color='lightgreen')
    plt.title('Top 15 Features by Importance')
    plt.xlabel('Importance Value')
    
    # Cross-validation scores
    plt.subplot(2, 2, 3)
    cv_scores = results['cv_scores']
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-')
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {cv_scores.mean():.4f}')
    plt.xlabel('CV Fold')
    plt.ylabel('AUC Score')
    plt.title('Cross-Validation AUC Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # ROC Curve
    plt.subplot(2, 2, 4)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, 'b-', label=f'ROC Curve (AUC = {results["test_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('random_forest_misjudgement_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main execution function."""
    print("=== Random Forest Classifier for Misjudgement Prediction ===\n")
    
    # Paths
    csv_reports_path = "CSV_Reports"
    json_file_path = "CodeJudge_Eval_0shot_easy_c_with_locations_with_evaluation_4o-mini_mis.json"
    
    # Load data
    print("Loading CSV metrics data...")
    csv_data = load_csv_data(csv_reports_path)
    
    print("\nLoading misjudgement data...")
    json_df = load_misjudgement_data(json_file_path)
    
    print("\nMerging all data sources...")
    merged_df = merge_all_data(csv_data, json_df)
    
    print("\nPreparing features and target...")
    X, y, feature_names = prepare_features_and_target(merged_df)
    
    if len(X) == 0:
        print("ERROR: No matching records found between CSV and JSON data!")
        return
    
    print("\nTraining Random Forest classifier...")
    model, scaler, feature_importance, results = train_random_forest(X, y, feature_names)
    
    print("\nGenerating plots...")
    plot_results(feature_importance, results, results['y_test'], results['y_pred_proba'])
    
    # Save the trained model and results
    print("\nSaving results...")
    merged_df.to_csv('misjudgement_dataset.csv', index=False)
    feature_importance.to_csv('feature_importance.csv', index=False)
    
    print("\n=== Training Complete ===")
    print(f"Best Test AUC Score: {results['test_auc']:.4f}")
    print(f"Mean CV AUC Score: {results['cv_scores'].mean():.4f}")
    print("\nFiles saved:")
    print("- misjudgement_dataset.csv: Complete dataset with features and target")
    print("- feature_importance.csv: Feature importance rankings")
    print("- random_forest_misjudgement_results.png: Visualization plots")


if __name__ == "__main__":
    main()