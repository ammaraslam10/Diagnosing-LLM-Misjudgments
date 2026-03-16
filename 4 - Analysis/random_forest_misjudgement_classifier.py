#!/usr/bin/env python3
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
import shap
import warnings
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
warnings.filterwarnings('ignore')

# Set up plotting parameters
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


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
        pylint_symbols = pylint_df.groupby(['module', 'type']).size().unstack(fill_value=0)
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
            'Complexity': ['sum']
        }).round(2)
        complexipy_agg.columns = ['complexity_sum']
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
            'data_id': item.get('data_id', 0),
            "difficulty": 0 if item.get('difficulty') == 'introductory' else 1 if item.get('difficulty') == 'interview' else 2 if item.get('difficulty') == 'competition' else None,
            "problem_text_length": item.get('problem_text_length', 0),
            "solution_text_length": item.get('solution_text_length', 0),
            "prompt_perplexity": item.get('prompt_perplexity', 0),
            "statement_gunning_fog_index": item.get('statement_gunning_fog_index', 0),
            "statement_flesch_kincaid_grade": item.get('statement_flesch_kincaid_grade', 0),
            "api_calls": item.get('api_calls', 0),
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
    
    # Fill remaining missing values with median
    for col in numeric_columns:
        if merged_df[col].isna().any():
            missing_count = merged_df[col].isna().sum()
            print(f"Missing data in {col}: {missing_count} values ({missing_count/len(merged_df)*100:.1f}%) - filling with median")
            merged_df[col] = merged_df[col].fillna(merged_df[col].median())
    
    # Keep identification columns for detailed analysis
    identification_columns = ['filename_base', 'task_id', 'source', 'answer', 'llm_answer', 'evaluated', 'data_id']
    
    return merged_df, identification_columns


def prepare_features_and_target(df, identification_columns):
    """Prepare feature matrix and target variable."""
    # Define feature renaming dictionary for better readability
    feature_rename_dict = {
        # Radon metrics
        'LOC': 'Lines of Code',
        'LLOC': 'Logical Lines of Code',
        'SLOC': 'Source Lines of Code',
        'Comments': 'Comment Lines',
        'Cyclomatic Complexity': 'Cyclomatic Complexity',
        'Maintainability Index': 'Maintainability Index',
        'h1': 'Unique Operators',
        'h2': 'Unique Operands', 
        'h': 'Vocabulary Size',
        'N1': 'Total Operators',
        'N2': 'Total Operands',
        'N': 'Program Length',
        'Vocabulary': 'Vocabulary',
        'Volume': 'Halstead Volume',
        'Difficulty': 'Halstead Difficulty',
        'Effort': 'Halstead Effort',
        'Bugs': 'Halstead Bugs',
        'Time': 'Halstead Time',
        
        # Pylint metrics
        'pylint_total_issues': 'Total Pylint Issues',
        'pylint_convention': 'Pylint Convention Issues',
        'pylint_refactor': 'Pylint Refactor Issues',
        'pylint_warning': 'Pylint Warning Issues',
        'pylint_error': 'Pylint Error Issues',

        # Complexipy metrics
        'complexity_sum': 'Cognitive Complexity',
        
        # Bandit security metrics
        'bandit_total_issues': 'Total Security Issues',
        'bandit_unique_severities': 'Security Severity Types',
        'bandit_unique_confidences': 'Security Confidence Types',
        
        # Problem characteristics
        'difficulty': 'Problem Difficulty Level',
        'problem_text_length': 'Problem Description Length',
        'solution_text_length': 'Solution Code Length', 
        "prompt_perplexity": "Prompt Perplexity",
        "statement_gunning_fog_index": "Statement Gunning Fog Index",
        "statement_flesch_kincaid_grade": "Statement Flesch-Kincaid Grade",
        "api_calls": "Function Calls"
    }
    
    # Define base feature columns (all numeric metrics)
    base_feature_columns = [
        # Radon metrics
        'LOC', 'LLOC', #'SLOC',
        'Comments', 'Cyclomatic Complexity', 
        'Maintainability Index', #'h1', 'h2', 'h', 'N1', 'N2', 'N', 
        #'Vocabulary', 
        'Volume', 'Difficulty', 'Effort', 'Bugs', 'Time',
        
        # Pylint total issues
        # 'pylint_total_issues',

        # Complexipy metrics
        'complexity_sum',
        
        # Bandit metrics
        'bandit_total_issues', 'bandit_unique_severities', 'bandit_unique_confidences',

        "difficulty", "problem_text_length", "solution_text_length", "prompt_perplexity",
        "statement_gunning_fog_index", "statement_flesch_kincaid_grade", "api_calls"
    ]
    
    # Add all pylint symbol columns (they start with 'pylint_' and are not 'pylint_total_issues')
    pylint_symbol_columns = [col for col in df.columns if col.startswith('pylint_') and col != 'pylint_total_issues']
    
    # Combine base features and pylint symbols (no missingness indicators yet - they'll be added during training)
    feature_columns = base_feature_columns + pylint_symbol_columns
    
    # Filter columns that actually exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    print(f"Available features ({len(available_features)}): {len(pylint_symbol_columns)} pylint symbols + {len(base_feature_columns)} base metrics")
    print(f"Pylint symbols found: {pylint_symbol_columns[:10]}{'...' if len(pylint_symbol_columns) > 10 else ''}")
    
    # Create a copy of the dataframe and rename columns using the dictionary
    df_renamed = df.copy()
    
    # Only rename columns that exist in both the dataframe and our rename dictionary
    columns_to_rename = {old_name: new_name for old_name, new_name in feature_rename_dict.items() 
                        if old_name in df_renamed.columns}
    
    print(f"Renaming {len(columns_to_rename)} features with descriptive names")
    df_renamed = df_renamed.rename(columns=columns_to_rename)
    
    # Update available_features to use renamed column names
    renamed_features = [feature_rename_dict.get(col, col) for col in available_features]
    
    # Separate features and identification info
    X = df_renamed[renamed_features]
    y = df['misjudgement']
    
    # Keep identification data for later use
    identification_data = df[identification_columns]
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable shape: {y.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Sample renamed features: {renamed_features[:5]}")
    
    return X, y, renamed_features, identification_data

def train_random_forest_with_shap(X, y, feature_names, identification_data):
    """Train Random Forest classifier and perform SHAP analysis."""
    # Split the data while keeping identification info aligned
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, identification_data, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Hyperparameter tuning with reduced grid for faster execution
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 15],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    
    # Create and tune Random Forest classifier
    rf = RandomForestClassifier(random_state=42)
    print("Performing hyperparameter tuning...")
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='roc_auc', 
        n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Make predictions
    y_pred = best_rf.predict(X_test)
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]
    
    # Evaluate the model
    print("\nModel Evaluation")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {auc_score:.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # SHAP Analysis
    print("\nSHAP Analysis")
    print("Computing SHAP values (this may take a few moments)...")
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_test)
    
    # For binary classification, get SHAP values for positive class
    print(f"SHAP values type: {type(shap_values)}")
    print(f"SHAP values shape: {np.array(shap_values).shape if hasattr(shap_values, 'shape') else 'no shape attr'}")
    
    if isinstance(shap_values, list) and len(shap_values) == 2:
        # Traditional format: list of arrays for each class
        shap_values_pos = shap_values[1]  # SHAP values for misjudgement = True
        shap_vals_for_plots = shap_values[1]
        print(f"Using list format - positive class shape: {shap_values_pos.shape}")
    elif hasattr(shap_values, 'shape') and len(shap_values.shape) == 3:
        # New format: single array with shape (n_samples, n_features, n_classes)
        shap_values_pos = shap_values[:, :, 1]  # Get positive class
        shap_vals_for_plots = shap_values[:, :, 1]
        print(f"Using 3D format - extracted positive class shape: {shap_values_pos.shape}")
    else:
        # Single class or other format
        shap_values_pos = shap_values
        shap_vals_for_plots = shap_values
        print(f"Using single format - shape: {shap_values_pos.shape}")
    
    print(f"Final SHAP values shape for analysis: {shap_values_pos.shape}")
    print(f"Number of features: {len(feature_names)}")
    
    # Calculate mean absolute SHAP values for feature importance
    mean_shap_values = np.abs(shap_values_pos).mean(axis=0)
    
    # Ensure mean_shap_values is 1D
    if mean_shap_values.ndim > 1:
        mean_shap_values = mean_shap_values.flatten()
    
    print(f"Mean SHAP values shape: {mean_shap_values.shape}")
    print(f"Feature names length: {len(feature_names)}")
    
    # Only use the first len(feature_names) values if there's a mismatch
    if len(mean_shap_values) > len(feature_names):
        mean_shap_values = mean_shap_values[:len(feature_names)]
    elif len(mean_shap_values) < len(feature_names):
        feature_names = feature_names[:len(mean_shap_values)]
    
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'shap_importance': mean_shap_values
    }).sort_values('shap_importance', ascending=False)
    
    print("\n=== Features by SHAP Importance ===")
    print(shap_importance)
    
    # Traditional feature importance for comparison
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'rf_importance': best_rf.feature_importances_
    }).sort_values('rf_importance', ascending=False)
    
    # Combine both importance measures
    combined_importance = shap_importance.merge(feature_importance, on='feature')
    
    # Cross-validation scores
    cv_scores = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"\nCross-validation AUC scores: {cv_scores}")
    print(f"Mean CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return best_rf, explainer, shap_vals_for_plots, combined_importance, {
        'test_auc': auc_score,
        'cv_scores': cv_scores,
        'best_params': grid_search.best_params_,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred_proba': y_pred_proba
    }


def create_shap_visualizations(explainer, shap_values, X_test, feature_names, combined_importance, results):
    """Create comprehensive SHAP visualizations as separate, readable plots."""
    print("Creating individual SHAP visualizations...")
    
    # Handle different SHAP value formats
    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_vals_for_plots = shap_values[1]  # Use positive class for binary classification
    else:
        shap_vals_for_plots = shap_values
    
    top_features = combined_importance.head(15)
    
    # Create clustering for feature correlation grouping
    print("Computing feature clustering based on correlations...")
    try:
        # Compute feature clustering using SHAP's hierarchical clustering
        clustering = shap.utils.hclust(X_test, y=results['y_test'])
        print(f"Successfully computed clustering with {len(clustering)} groups")

        print("Z shape:", clustering.shape)
        print("First 10 merges:\n", pd.DataFrame(clustering[:10], columns=["c1","c2","dist","count"]))

        from scipy.cluster.hierarchy import fcluster, leaves_list
        def shap_feature_groups(clustering, feature_names, cutoff=0.7, keep_singletons=False):
            labels = fcluster(clustering, t=cutoff, criterion="distance")
            df = pd.DataFrame({"feature": feature_names, "group": labels})
            groups = (df.groupby("group")["feature"]
                        .apply(list)
                        .sort_values(key=lambda s: s.apply(len), ascending=False))
            if not keep_singletons:
                groups = groups[groups.apply(len) > 1]
            return groups  # pandas Series: index=group_id, value=[features...]

        # Use SAME cutoff you use in shap.plots.bar(..., clustering_cutoff=...)
        groups = shap_feature_groups(clustering, X_test.columns, cutoff=0.3, keep_singletons=False)

        # Print groups + items
        for gid, items in groups.items():
            print(f"Group {gid} (n={len(items)}): {items}")
    except Exception as e:
        print(f"Error computing clustering: {e}")
        clustering = None
    
    # Create SHAP Explanation object for newer SHAP versions
    try:
        if isinstance(shap_vals_for_plots, np.ndarray):
            # Create Explanation object with proper structure
            explanation = shap.Explanation(values=shap_vals_for_plots, 
                                         base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, np.ndarray) else explainer.expected_value,
                                         data=X_test.values, 
                                         feature_names=feature_names)
            print("Successfully created SHAP Explanation object")
        else:
            explanation = shap_vals_for_plots
            print("Using existing SHAP Explanation object")
    except Exception as e:
        print(f"Error creating SHAP Explanation object: {e}")
        explanation = None
    
    # Plot 1: SHAP Summary Plot (Beeswarm)
    print("Creating SHAP beeswarm plot...")
    plt.figure(figsize=(12, 8))
    try:
        shap.summary_plot(shap_vals_for_plots, X_test, feature_names=feature_names, show=False, max_display=15)
        plt.title('SHAP Feature Importance - Impact on Misjudgement Prediction', fontsize=14, pad=20)
        plt.xlabel('SHAP Value (impact on model output)', fontsize=12)
        plt.tight_layout()
        plt.savefig('report/1_shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating beeswarm plot: {e}")
        plt.figure(figsize=(12, 8))
        plt.text(0.5, 0.5, f'SHAP Beeswarm Plot\nError: {str(e)}', ha='center', va='center', fontsize=12)
        plt.title('SHAP Feature Importance (Beeswarm) - Error')
        plt.savefig('report/1_shap_beeswarm_plot_error.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 2: SHAP Summary Plot (Bar) with clustering
    print("Creating SHAP bar plot with clustering...")
    plt.figure(figsize=(12, 8))
    try:
        if clustering is not None and explanation is not None:
            shap.plots.bar(explanation, clustering=clustering, max_display=20, show=False, clustering_cutoff=0.3)
            plt.title('SHAP Feature Importance - Clustered by Correlation', fontsize=14, pad=20)
        else:
            shap.summary_plot(shap_vals_for_plots, X_test, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
            plt.title('SHAP Feature Importance - Mean Absolute Impact', fontsize=14, pad=20)
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.tight_layout()
        plt.savefig('report/2_shap_bar_plot_clustered.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error creating clustered bar plot: {e}")
        # Fallback to regular bar plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals_for_plots, X_test, feature_names=feature_names, plot_type="bar", show=False, max_display=15)
        plt.title('SHAP Feature Importance - Mean Absolute Impact (Fallback)', fontsize=14, pad=20)
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.tight_layout()
        plt.savefig('report/2_shap_bar_plot_fallback.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Plot 4: SHAP vs Random Forest Importance Comparison
    print("Creating SHAP vs RF importance comparison...")
    plt.figure(figsize=(14, 10))
    
    x_pos = np.arange(len(top_features))
    width = 0.35
    
    bars1 = plt.barh(x_pos - width/2, top_features['shap_importance'], width, 
                     label='SHAP Importance', alpha=0.8, color='skyblue')
    bars2 = plt.barh(x_pos + width/2, top_features['rf_importance'], width, 
                     label='Random Forest Importance', alpha=0.8, color='lightcoral')
    
    plt.yticks(x_pos, top_features['feature'], fontsize=10)
    plt.xlabel('Importance Score', fontsize=12)
    plt.title('Feature Importance Comparison: SHAP vs Random Forest', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/4_shap_vs_rf_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 5: ROC Curve
    print("Creating ROC curve...")
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(results['y_test'], results['y_pred_proba'])
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {results["test_auc"]:.4f})')
    plt.plot([0, 1], [0, 1], 'r--', alpha=0.8, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/5_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 6: Cross-validation scores
    print("Creating cross-validation scores plot...")
    plt.figure(figsize=(10, 6))
    cv_scores = results['cv_scores']
    plt.plot(range(1, len(cv_scores) + 1), cv_scores, 'bo-', markersize=8, linewidth=2)
    plt.axhline(y=cv_scores.mean(), color='r', linestyle='--', linewidth=2, 
                label=f'Mean: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')
    plt.fill_between(range(1, len(cv_scores) + 1), 
                     cv_scores.mean() - cv_scores.std(), 
                     cv_scores.mean() + cv_scores.std(), 
                     alpha=0.2, color='red')
    plt.xlabel('Cross-Validation Fold', fontsize=12)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('Cross-Validation Performance Stability', fontsize=14, pad=20)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim([min(cv_scores) - 0.05, max(cv_scores) + 0.05])
    plt.tight_layout()
    plt.savefig('report/6_cross_validation_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 7: Feature categories breakdown
    print("Creating feature categories pie chart...")
    plt.figure(figsize=(10, 8))
    
    # Categorize features
    categories = {
        'Radon Complexity': [],
        'Pylint Issues': [],
        'Function Complexity': [],
        'Security Issues': []
    }
    
    for _, row in top_features.head(10).iterrows():
        feature = row['feature']
        if feature.startswith('pylint_'):
            categories['Pylint Issues'].append(row['shap_importance'])
        elif feature in ['complexity_max', 'complexity_sum']:
            categories['Function Complexity'].append(row['shap_importance'])
        elif feature.startswith('bandit_'):
            categories['Security Issues'].append(row['shap_importance'])
        else:
            categories['Radon Complexity'].append(row['shap_importance'])
    
    # Sum importance by category
    category_importance = {k: sum(v) for k, v in categories.items() if v}
    
    if category_importance:
        colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        wedges, texts, autotexts = plt.pie(category_importance.values(), 
                                           labels=category_importance.keys(), 
                                           autopct='%1.1f%%',
                                           colors=colors,
                                           explode=[0.05] * len(category_importance),
                                           shadow=True,
                                           startangle=90)
        
        # Enhance text appearance
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(11)
        
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        plt.title('SHAP Importance Distribution by Feature Category\n(Top 10 Features)', 
                  fontsize=14, pad=20, fontweight='bold')
    else:
        plt.text(0.5, 0.5, 'No categorical data available', ha='center', va='center', 
                fontsize=14, fontweight='bold')
        plt.title('Feature Categories - No Data Available', fontsize=14, pad=20)
    
    plt.tight_layout()
    plt.savefig('report/7_feature_categories_pie.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 8: Confusion Matrix Heatmap
    print("Creating confusion matrix heatmap...")
    plt.figure(figsize=(8, 6))
    
    # Create confusion matrix from test results
    cm = confusion_matrix(results['y_test'], (results['y_pred_proba'] > 0.5).astype(int))
    
    # Create heatmap with seaborn
    ax = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                     xticklabels=['Misjudged is false', 'Misjudged is true'], 
                     yticklabels=['Misjudged is false', 'Misjudged is true'],
                     cbar_kws={'label': 'Number of Cases'},
                     annot_kws={'size': 16})
    
    # Increase font sizes
    plt.title('Confusion Matrix - Misjudgement Prediction', fontsize=18, pad=20)
    plt.xlabel('Predicted Label', fontsize=16)
    plt.ylabel('True Label', fontsize=16)
    
    # Increase tick label font sizes
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Increase colorbar font size
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label('Number of Cases', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('report/8_confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 9: Detailed SHAP importance with color coding
    print("Creating detailed SHAP importance plot...")
    plt.figure(figsize=(14, 10))
    
    top_20_features = combined_importance.head(20)
    
    # Color bars based on feature category
    colors = []
    for feature in top_20_features['feature']:
        if feature.startswith('pylint_'):
            colors.append('#e74c3c')  # Red for Pylint issues
        elif feature in ['complexity_max', 'complexity_sum']:
            colors.append('#f39c12')  # Orange for function complexity
        elif feature.startswith('bandit_'):
            colors.append('#9b59b6')  # Purple for security issues
        else:
            colors.append('#3498db')  # Blue for Radon complexity
    
    bars = plt.barh(range(len(top_20_features)), top_20_features['shap_importance'], color=colors, alpha=0.8)
    plt.yticks(range(len(top_20_features)), top_20_features['feature'], fontsize=10)
    plt.xlabel('SHAP Importance (Mean Absolute Value)', fontsize=12)
    plt.title('Top 20 Features by SHAP Importance\n(Color-coded by Category)', fontsize=14, pad=20)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_20_features['shap_importance'])):
        plt.text(value + 0.001, i, f'{value:.4f}', va='center', fontsize=9)
    
    # Create legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, color='#3498db', alpha=0.8, label='Radon Complexity'),
        plt.Rectangle((0,0),1,1, color='#e74c3c', alpha=0.8, label='Pylint Issues'),
        plt.Rectangle((0,0),1,1, color='#f39c12', alpha=0.8, label='Function Complexity'),
        plt.Rectangle((0,0),1,1, color='#9b59b6', alpha=0.8, label='Security Issues')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('report/9_detailed_shap_importance.png', dpi=300, bbox_inches='tight')
    plt.close()


def analyze_feature_effects(combined_importance, X, y):
    """Analyze the effects of top features on misjudgement."""
    print("\n=== Feature Effect Analysis ===")
    
    top_features = combined_importance.head(10)['feature'].tolist()
    
    print("Top 10 features and their relationship with misjudgement:")
    
    analysis_results = []
    
    for feature in top_features:
        if feature in X.columns:
            # Calculate correlation with target
            correlation = X[feature].corr(y.astype(int))
            
            # Calculate mean values for misjudged vs correctly judged
            mean_misjudged = X[y == True][feature].mean()
            mean_correct = X[y == False][feature].mean()
            
            # Calculate effect direction
            effect = "increases" if mean_misjudged > mean_correct else "decreases"
            
            analysis_results.append({
                'feature': feature,
                'correlation': correlation,
                'mean_misjudged': mean_misjudged,
                'mean_correct': mean_correct,
                'effect': effect,
                'difference': abs(mean_misjudged - mean_correct)
            })
            
            print(f"{feature}:")
            print(f"  - Correlation with misjudgement: {correlation:.4f}")
            print(f"  - Mean (misjudged cases): {mean_misjudged:.2f}")
            print(f"  - Mean (correct cases): {mean_correct:.2f}")
            print(f"  - Effect: Higher values {effect} misjudgement likelihood")
            print()
    
    return analysis_results


def main():
    """Main execution function."""
    print("Random Forest Classifier with SHAP Analysis\n")
    
    # Paths
    csv_reports_path = "CSV_Reports"
    json_file_path = "CodeJudge_Eval_0shot_easy_c_with_locations_with_evaluation.json"
    
    # Load data
    print("Loading CSV metrics data...")
    csv_data = load_csv_data(csv_reports_path)
    
    print("\nLoading misjudgement data...")
    json_df = load_misjudgement_data(json_file_path)
    
    print("\nMerging all data sources...")
    merged_df, identification_columns = merge_all_data(csv_data, json_df)
    
    print("\nPreparing features and target...")
    X, y, feature_names, identification_data = prepare_features_and_target(merged_df, identification_columns)
    
    if len(X) == 0:
        print("ERROR: No matching records found between CSV and JSON data!")
        return
    
    print("\nTraining Random Forest classifier with SHAP analysis...")
    model, explainer, shap_values, combined_importance, results = train_random_forest_with_shap(X, y, feature_names, identification_data)
    
    print("\nCreating SHAP visualizations...")
    create_shap_visualizations(explainer, shap_values, results['X_test'], feature_names, combined_importance, results)
    
    print("\nAnalyzing feature effects...")
    feature_effects = analyze_feature_effects(combined_importance, X, y)
    
    # Save results
    print("\nSaving results...")
    merged_df.to_csv('report/misjudgement_dataset_with_shap.csv', index=False)
    combined_importance.to_csv('report/shap_and_rf_feature_importance.csv', index=False)
    
    # Create summary report
    summary_report = {
        'model_performance': {
            'test_auc': results['test_auc'],
            'mean_cv_auc': results['cv_scores'].mean(),
            'best_params': results['best_params']
        },
        'top_shap_features': combined_importance.head(10).to_dict('records'),
        'feature_effects': feature_effects
    }
    
    with open('report/shap_analysis_summary.json', 'w') as f:
        json.dump(summary_report, f, indent=2, default=str)
    
    print("\nSHAP Analysis Complete")
    print(f"Best Test AUC Score: {results['test_auc']:.4f}")
    print(f"Mean CV AUC Score: {results['cv_scores'].mean():.4f}")

    print("\nTOP FEATURES INFLUENCING LLM MISJUDGMENTS (SHAP Analysis)")
    print(combined_importance.head(100)[['feature', 'shap_importance', 'rf_importance']])


if __name__ == "__main__":
    main()