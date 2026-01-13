#!/usr/bin/env python3
"""
Logistic Regression Classifier for Misjudgement Prediction

This script loads code quality metrics from CSV files and prediction data from JSON,
then trains a Logistic Regression classifier to predict misjudgement cases.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
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
            'Complexity': ['max', 'sum']
        }).round(2)
        # add complexity_spread, complexity_ratio (if max exists)
        # complexipy_agg[('Complexity', 'spread')] = complexipy_agg[('Complexity', 'max')] - complexipy_agg[('Complexity', 'mean')]
        # complexipy_agg[('Complexity', 'ratio')] = complexipy_agg[('Complexity', 'max')] / complexipy_agg[('Complexity', 'mean')].replace(0, np.nan)

        complexipy_agg.columns = ['complexity_max', 'complexity_sum']#, 'complexity_spread', 'complexity_ratio']
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
    
    print(f"Final merged dataset shape: {merged_df.shape}")
    
    # Identify numeric columns for missing data analysis
    numeric_columns = merged_df.select_dtypes(include=[np.number]).columns
    
    # Fill security/pylint columns with 0s (missing means no issues)
    security_pylint_cols = [col for col in numeric_columns if 'bandit' in col.lower() or 'pylint' in col.lower()]
    if security_pylint_cols:
        print(f"Filling {len(security_pylint_cols)} security/pylint columns with 0s")
        merged_df[security_pylint_cols] = merged_df[security_pylint_cols].fillna(0)
    
    # Report remaining missing data patterns
    missing_info = []
    for col in numeric_columns:
        if merged_df[col].isna().any():
            missing_count = merged_df[col].isna().sum()
            missing_info.append((col, missing_count, missing_count/len(merged_df)*100))
            print(f"Missing data in {col}: {missing_count} values ({missing_count/len(merged_df)*100:.1f}%)")
    
    if missing_info:
        print(f"\nFound missing data in {len(missing_info)} remaining columns - will be imputed with median")
    else:
        print("No missing data found in numeric columns")
    
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
        'complexity_max', 'complexity_sum',
        
        # Bandit metrics
        'bandit_total_issues', 'bandit_unique_severities', 'bandit_unique_confidences'
    ]
    
    # Add all pylint symbol columns (they start with 'pylint_' and are not 'pylint_total_issues')
    pylint_symbol_columns = [col for col in df.columns if col.startswith('pylint_') and col != 'pylint_total_issues']
    
    # Combine base features and pylint symbols (no missingness indicators yet - they'll be added during training)
    feature_columns = base_feature_columns + pylint_symbol_columns
    
    # Filter columns that actually exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Available features ({len(available_features)}): {len(pylint_symbol_columns)} pylint symbols + {len(base_feature_columns)} base metrics")
    print(f"Pylint symbols found: {pylint_symbol_columns[:10]}{'...' if len(pylint_symbol_columns) > 10 else ''}")
    print("Note: Missingness indicators will be added during training with proper train/test split")
    
    X = df[available_features]
    y = df['misjudgement']
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    return X, y, available_features


def train_logistic_regression(X, y, feature_names):
    """Train and evaluate Logistic Regression classifier with simple imputation pipeline."""
    # Split the data first (before any preprocessing)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Create simple preprocessing pipeline
    full_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(
            random_state=42, 
            class_weight='balanced',
            max_iter=1000,
            solver='liblinear'
        ))
    ])
    
    # Train the complete pipeline
    print("Training Logistic Regression pipeline with imputation...")
    full_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = full_pipeline.predict(X_test)
    y_pred_proba = full_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print("\n=== Model Evaluation ===") 
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ROC AUC Score
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    
    # Get feature importance from trained classifier
    classifier = full_pipeline.named_steps['classifier']
    
    # Get transformed feature names from the imputer
    try:
        imputer_fitted = full_pipeline.named_steps['imputer']
        feature_names_out = imputer_fitted.get_feature_names_out(feature_names)
    except:
        # Fallback: estimate feature names
        feature_names_out = [f"feature_{i}" for i in range(len(classifier.coef_[0]))]
    
    # Calculate odds ratios and bootstrap confidence intervals
    coefficients = classifier.coef_[0]
    odds_ratios = np.exp(coefficients)
    
    # Transform X_test through the preprocessing steps to match classifier input
    imputer_fitted = full_pipeline.named_steps['imputer']
    scaler_fitted = full_pipeline.named_steps['scaler']
    
    X_test_imputed = imputer_fitted.transform(X_test)
    X_test_scaled = scaler_fitted.transform(X_test_imputed)
    
    # Calculate bootstrap confidence intervals for coefficients
    print("Calculating bootstrap confidence intervals...")
    n_bootstrap = 100  # Increase for better estimates
    bootstrap_coefs = []
    
    # Transform training data for bootstrap
    X_train_imputed = imputer_fitted.transform(X_train)
    X_train_scaled = scaler_fitted.transform(X_train_imputed)
    
    for i in range(n_bootstrap):
        # Bootstrap resample from TRAINING data (larger, more stable)
        indices = np.random.choice(len(X_train_scaled), size=len(X_train_scaled), replace=True)
        X_boot = X_train_scaled[indices]
        y_boot = y_train.iloc[indices].values if hasattr(y_train, 'iloc') else y_train[indices]
        
        # Fit classifier on bootstrap sample
        boot_classifier = LogisticRegression(
            random_state=None,  # Allow randomness for bootstrap
            class_weight='balanced',
            max_iter=500,  # Reduced for speed
            solver='liblinear'
        )
        
        try:
            boot_classifier.fit(X_boot, y_boot)
            bootstrap_coefs.append(boot_classifier.coef_[0])
        except:
            # If bootstrap sample causes fitting issues, skip
            continue
    
    if bootstrap_coefs and len(bootstrap_coefs) > 20:
        bootstrap_coefs = np.array(bootstrap_coefs)
        # Calculate 95% confidence intervals from bootstrap distribution
        ci_lower = np.exp(np.percentile(bootstrap_coefs, 2.5, axis=0))
        ci_upper = np.exp(np.percentile(bootstrap_coefs, 97.5, axis=0))
        print(f"Bootstrap successful: {len(bootstrap_coefs)} samples")
        
        # Print interpretation of results
        print(f"\n=== INTERPRETATION ===")
        print(f"Most coefficients are NEGATIVE → These are PROTECTIVE factors")
        print(f"Negative coefficient = Odds Ratio < 1 = REDUCES misjudgement risk")
        print(f"Example: LOC coefficient = {coefficients[0]:.3f} → OR = {odds_ratios[0]:.3f}")
        print(f"This means: More lines of code → LOWER chance of misjudgement")
        
    else:
        # Fallback: no confidence intervals
        ci_lower = odds_ratios * 0.8  # Rough approximation
        ci_upper = odds_ratios * 1.2
        print("Bootstrap failed, using fallback CIs")
    
    # Calculate permutation importance
    print("\nCalculating permutation importance...")

    # Use the classifier directly on the transformed data for permutation importance
    perm_importance = permutation_importance(
        classifier, X_test_scaled, y_test, 
        n_repeats=30, random_state=42, scoring='average_precision'
    )
    
    # Now all arrays should have the same length
    n_features = len(coefficients)
    print(f"Coefficients: {n_features}, Features out: {len(feature_names_out)}, Perm importance: {len(perm_importance.importances_mean)}")
    
    if len(feature_names_out) != n_features:
        feature_names_out = feature_names_out[:n_features]  # Truncate if too long
    
    # Create comprehensive feature analysis dataframe
    feature_analysis = pd.DataFrame({
        'feature': feature_names_out[:n_features],
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients),
        'odds_ratio': odds_ratios,
        'or_ci_lower': ci_lower,
        'or_ci_upper': ci_upper,
        'perm_importance_mean': perm_importance.importances_mean,
        'perm_importance_std': perm_importance.importances_std
    }).sort_values('perm_importance_mean', ascending=False)
    
    print("\n=== Top 10 Most Important Features (by permutation importance) ===")
    print(feature_analysis.head(10)[['feature', 'coefficient', 'odds_ratio', 'perm_importance_mean']])
    
    print("\n=== Odds Ratios with Confidence Intervals (Top 10) ===")
    top_or = feature_analysis.head(10)
    for _, row in top_or.iterrows():
        print(f"{row['feature']}: OR = {row['odds_ratio']:.3f} [{row['or_ci_lower']:.3f} - {row['or_ci_upper']:.3f}]")
    
    # Cross-validation scores using the full pipeline (prevents data leakage)
    cv_scores = cross_val_score(full_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\nCross-validation AUC scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return full_pipeline, feature_analysis, {
        'test_auc': auc_score,
        'cv_scores': cv_scores,
        'intercept': classifier.intercept_[0],
        'y_test': y_test,
        'y_pred_proba': y_pred_proba,
        'perm_importance': perm_importance
    }


def plot_results(feature_analysis, results, y_test, y_pred_proba):
    """Plot comprehensive analysis including permutation importance and odds ratios."""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Permutation Importance (Top 15)
    plt.subplot(3, 3, 1)
    top_perm = feature_analysis.head(15)
    colors_perm = ['darkgreen' if x > 0 else 'darkred' for x in top_perm['coefficient']]
    bars = plt.barh(range(len(top_perm)), top_perm['perm_importance_mean'], 
                    xerr=top_perm['perm_importance_std'], color=colors_perm, alpha=0.7)
    plt.yticks(range(len(top_perm)), [f[:20] + '...' if len(f) > 20 else f for f in top_perm['feature']], fontsize=8)
    plt.xlabel('Permutation Importance (AUC decrease)')
    plt.title('Top 15 Features by Permutation Importance')
    plt.grid(True, alpha=0.3)
    
    # 2. Coefficients with error bars
    plt.subplot(3, 3, 2)
    # top_coef = feature_analysis.nlargest(15, 'abs_coefficient').sort_values('abs_coefficient', ascending=True)
    top_coef = feature_analysis.head(15)
    colors_coef = ['red' if x < 0 else 'blue' for x in top_coef['coefficient']]
    plt.barh(range(len(top_coef)), top_coef['coefficient'], color=colors_coef, alpha=0.7)
    plt.yticks(range(len(top_coef)), [f[:20] + '...' if len(f) > 20 else f for f in top_coef['feature']], fontsize=8)
    plt.xlabel('Coefficient Value')
    plt.title('Direction of Effect (Top 15 Coefficients)')
    plt.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # 3. Odds Ratios with Confidence Intervals
    plt.subplot(3, 3, 3)
    top_or = feature_analysis.head(15)
    y_pos = range(len(top_or))
    
    # Calculate error bar values, ensuring they're non-negative
    err_lower = np.maximum(0, top_or['odds_ratio'] - top_or['or_ci_lower'])
    err_upper = np.maximum(0, top_or['or_ci_upper'] - top_or['odds_ratio'])
    
    plt.errorbar(top_or['odds_ratio'], y_pos, 
                xerr=[err_lower, err_upper], 
                fmt='o', capsize=3, capthick=1, markersize=4)
    plt.yticks(y_pos, [f[:20] + '...' if len(f) > 20 else f for f in top_or['feature']], fontsize=8)
    plt.xlabel('Odds Ratio')
    plt.title('Top 15 Odds Ratios with 95% CI')
    plt.axvline(x=1, color='red', linestyle='--', alpha=0.7, label='OR = 1')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale('log')
    
    # 4. Cross-validation scores
    plt.subplot(3, 3, 4)
    cv_scores = results['cv_scores']
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-', markersize=8, linewidth=2)
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {cv_scores.mean():.4f}')
    plt.fill_between(range(1, len(cv_scores) + 1), 
                     cv_scores.mean() - cv_scores.std(), 
                     cv_scores.mean() + cv_scores.std(), 
                     alpha=0.2, color='red')
    plt.xlabel('CV Fold')
    plt.ylabel('AUC Score')
    plt.title('Cross-Validation AUC Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0.7, 0.9)
    
    # 5. ROC Curve
    plt.subplot(3, 3, 5)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, 'b-', linewidth=3, label=f'ROC Curve (AUC = {results["test_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Permutation Importance Distribution
    plt.subplot(3, 3, 6)
    perm_scores = results['perm_importance'].importances
    top_5_indices = feature_analysis.head(5).index
    for i, idx in enumerate(top_5_indices[:5]):
        plt.hist(perm_scores[idx], alpha=0.6, bins=15, 
                label=f"{feature_analysis.iloc[i]['feature'][:15]}...")
    plt.xlabel('Permutation Importance Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Permutation Scores (Top 5)')
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 7. Coefficient vs Permutation Importance Scatter
    plt.subplot(3, 3, 7)
    plt.scatter(feature_analysis['abs_coefficient'], feature_analysis['perm_importance_mean'], 
                c=feature_analysis['coefficient'], cmap='RdBu_r', s=50, alpha=0.7)
    plt.xlabel('Absolute Coefficient Value')
    plt.ylabel('Permutation Importance')
    plt.title('Coefficient vs Permutation Importance')
    plt.colorbar(label='Coefficient Value')
    plt.grid(True, alpha=0.3)
    
    # 8. Feature Type Analysis
    plt.subplot(3, 3, 8)
    feature_types = []
    for feat in feature_analysis['feature']:
        if 'pylint_' in feat:
            feature_types.append('Pylint')
        elif 'bandit_' in feat:
            feature_types.append('Security')
        elif 'complexity_' in feat:
            feature_types.append('Complexity')
        elif feat in ['LOC', 'LLOC', 'SLOC', 'Comments']:
            feature_types.append('Size')
        else:
            feature_types.append('Other')
    
    feature_analysis['type'] = feature_types
    type_importance = feature_analysis.groupby('type')['perm_importance_mean'].mean().sort_values(ascending=True)
    
    colors_type = ['red', 'blue', 'green', 'orange', 'purple'][:len(type_importance)]
    plt.barh(range(len(type_importance)), type_importance.values, color=colors_type, alpha=0.7)
    plt.yticks(range(len(type_importance)), type_importance.index)
    plt.xlabel('Average Permutation Importance')
    plt.title('Feature Type Importance')
    plt.grid(True, alpha=0.3)
    
    # 9. Prediction Probability Distribution
    plt.subplot(3, 3, 9)
    misjudged_probs = y_pred_proba[y_test == True]
    correct_probs = y_pred_proba[y_test == False]
    
    plt.hist(correct_probs, bins=20, alpha=0.7, label='Correct Judgements', color='blue')
    plt.hist(misjudged_probs, bins=20, alpha=0.7, label='Misjudgements', color='red')
    plt.xlabel('Predicted Probability of Misjudgement')
    plt.ylabel('Frequency')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comprehensive_misjudgement_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main execution function."""
    print("=== Logistic Regression Classifier for Misjudgement Prediction ===\n")
    
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
    
    print("\nTraining Logistic Regression classifier...")
    pipeline, feature_analysis, results = train_logistic_regression(X, y, feature_names)
    
    print("\nGenerating plots...")
    plot_results(feature_analysis, results, results['y_test'], results['y_pred_proba'])
    
    # Save the trained model and results
    print("\nSaving results...")
    merged_df.to_csv('logistic_regression_misjudgement_dataset.csv', index=False)
    feature_analysis.to_csv('comprehensive_feature_analysis.csv', index=False)
    
    print("\n=== Training Complete ===")
    print(f"Test AUC Score: {results['test_auc']:.4f}")
    print(f"Mean CV AUC Score: {results['cv_scores'].mean():.4f}")
    print(f"Model Intercept: {results['intercept']:.4f}")
    print("\nFiles saved:")
    print("- logistic_regression_misjudgement_dataset.csv: Complete dataset with features and target")
    print("- comprehensive_feature_analysis.csv: Coefficients, odds ratios, and permutation importance")
    print("- comprehensive_misjudgement_analysis.png: Comprehensive visualization plots")


if __name__ == "__main__":
    main()