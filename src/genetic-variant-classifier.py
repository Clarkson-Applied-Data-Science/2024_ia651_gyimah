"""
Improved Genetic Variant Classification Framework with Proper Sequence Encoding

This script properly handles DNA sequence data by encoding it appropriately
for machine learning models, comparing population-aware and non-population-aware
approaches for genetic variant classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from typing import Dict, List, Tuple, Optional, Union, Any
import joblib
from datetime import datetime

# Data processing
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GroupKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, RobustScaler, QuantileTransformer, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import FeatureHasher

# Models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from fairlearn.reductions import GridSearch, DemographicParity, ErrorRate

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)


class Logger:
    """Class for handling logging throughout the application"""
    
    def __init__(self, level=logging.INFO):
        """Initialize the logger with specified level"""
        logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")
        self.logger = logging.getLogger(__name__)
    
    def info(self, message: str) -> None:
        """Log information message"""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log error message"""
        self.logger.error(message)


class DataLoader:
    """Class for loading and initial processing of genetic variant data"""
    
    def __init__(self, file_path: str, logger: Logger = None):
        """Initialize with file path and optional logger"""
        self.file_path = file_path
        self.logger = logger or Logger()
    
    def load_data(self) -> pd.DataFrame:
        """Load genetic variant data with appropriate dtypes"""
        self.logger.info(f"Loading data from {self.file_path}")
        
        # Optimized dtypes for memory usage
        dtype_mapping = {
            'gene': 'category',
            'chromosome': 'category',
            'variant_id': 'category',
            'consequence_type': 'category',
            'clinical_significance': 'category',
            'population_name': 'category',
            'region': 'category',
        }
        
        # Read the data
        df = pd.read_csv(self.file_path, dtype=dtype_mapping, low_memory=False)
        
        # Convert numeric columns safely
        numeric_cols = ['allele_freq', 'allele_count', 'sample_size', 'position']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.logger.info(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df


class DataPreprocessor:
    """Class for data preprocessing and feature engineering"""
    
    def __init__(self, logger: Logger = None):
        """Initialize with optional logger"""
        self.logger = logger or Logger()
        # Dictionary of available scalers
        self.scalers = {
            "none": None,
            "standard": StandardScaler(),
            "robust": RobustScaler(),
            "quantile": QuantileTransformer(output_distribution='normal')
        }
    @staticmethod
    def _concat_sequences(X):
        """
        X arrives as a DataFrame or 2-D ndarray containing the selected
        sequence columns.  Return a 1-D array-like of concatenated strings,
        one per sample.
        """
        if isinstance(X, pd.DataFrame):
            return X.astype(str).apply(''.join, axis=1)
        # ndarray fallback
        return np.array([''.join(map(str, row)) for row in X])
  
    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare the target variable (clinical significance)"""
        self.logger.info("Preparing target variable")
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Map clinical significance to three classes
        def map_to_classes(label_str):
            if pd.isna(label_str):
                return np.nan
            label = str(label_str).lower()
            if "pathogenic" in label:
                return "pathogenic"
            elif "benign" in label and "pathogenic" not in label:
                return "benign"
            elif "uncertain" in label or "conflicting" in label:
                return "uncertain"
            else:
                return np.nan
        
        # Create target column
        df["mapped_target"] = df["clinical_significance"].apply(map_to_classes)
        
        # Drop rows with undefined target
        df = df.dropna(subset=["mapped_target"])
        
        # Encode target as numeric
        label_map = {"benign": 0, "pathogenic": 1, "uncertain": 2}
        df["target_encoded"] = df["mapped_target"].map(label_map)
        
        # Show class distribution
        class_dist = df["mapped_target"].value_counts()
        self.logger.info(f"Target class distribution:\n{class_dist}")
        
        # Plot class distribution
        plt.figure(figsize=(8, 6))
        class_dist.plot(kind='bar')
        plt.title('Class Distribution')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig("class_distribution.png")
        
        return df
    
    def extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract numerical features from sequence data.
        This creates features that capture the biological information
        while being numeric and ML-friendly.
        """
        self.logger.info("Extracting sequence-based features")
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Handle the allele column if it exists
        if 'allele' in df.columns:
            # Replace NaN with empty string for feature extraction
            df['allele'] = df['allele'].fillna('').astype(str)
            
            # Basic length feature
            df['allele_length'] = df['allele'].str.len()
            
            # Nucleotide count features
            df['allele_A_count'] = df['allele'].str.count('A')
            df['allele_C_count'] = df['allele'].str.count('C')
            df['allele_G_count'] = df['allele'].str.count('G')
            df['allele_T_count'] = df['allele'].str.count('T')
            
            # GC content (informative for biological properties)
            df['allele_GC_content'] = (df['allele_G_count'] + df['allele_C_count']) / df['allele_length'].replace(0, 1)
        
        # Do the same for allele_string if present
        if 'allele_string' in df.columns and 'allele' not in df.columns:
            df['allele_string'] = df['allele_string'].fillna('').astype(str)
            df['allele_length'] = df['allele_string'].str.len()
            df['allele_A_count'] = df['allele_string'].str.count('A')
            df['allele_C_count'] = df['allele_string'].str.count('C')
            df['allele_G_count'] = df['allele_string'].str.count('G')
            df['allele_T_count'] = df['allele_string'].str.count('T')
            df['allele_GC_content'] = (df['allele_G_count'] + df['allele_C_count']) / df['allele_length'].replace(0, 1)
        
        sequence_features = df.columns[df.columns.str.contains('allele_')].tolist()
        self.logger.info(f"Extracted sequence features: {sequence_features}")
        return df
    
    def create_population_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create population-specific interaction features"""
        self.logger.info("Creating population-aware features")
        
        # Create a copy to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Population-consequence interaction
        if 'population_name' in df.columns and 'consequence_type' in df.columns:
            # Convert categorical to string before concatenation
            df['pop_consequence'] = df['population_name'].astype(str) + '_' + df['consequence_type'].astype(str)
        
        # Population-gene interaction
        if 'population_name' in df.columns and 'gene' in df.columns:
            # Convert categorical to string before concatenation
            df['pop_gene'] = df['population_name'].astype(str) + '_' + df['gene'].astype(str)
        
        # Population-allele frequency interaction
        if 'allele_freq' in df.columns and 'population_name' in df.columns:
            # Calculate mean allele frequency by population
            pop_freq_means = df.groupby(df['population_name'].astype(str))['allele_freq'].mean().to_dict()
            
            # Create a new column with the population mean
            df['pop_allele_freq_mean'] = df['population_name'].astype(str).map(pop_freq_means)
            
            # Handle missing values and calculate relative frequency
            mean_freq = df['allele_freq'].mean()
            df['pop_allele_freq_mean'] = df['pop_allele_freq_mean'].fillna(mean_freq)
            df['allele_freq_rel'] = df['allele_freq'] / df['pop_allele_freq_mean']
        
        return df
    
    def prepare_column_lists(self, df: pd.DataFrame, population_aware: bool = True) -> Tuple[List, List, List]:
        """Identify numeric, categorical, and sequence columns for preprocessing"""
        
        # Basic numeric features
        numeric_cols = [
            'allele_freq', 'allele_count', 'sample_size', 'position',
            'allele_length', 'allele_A_count', 'allele_C_count',
            'allele_G_count', 'allele_T_count', 'allele_GC_content'
        ]
        
        # Keep only columns that exist in the dataframe
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        # Population-specific numeric features (if population-aware)
        if population_aware and 'allele_freq_rel' in df.columns:
            numeric_cols.append('allele_freq_rel')
        
        # Basic categorical features
        categorical_cols = ['gene', 'chromosome', 'consequence_type', 'region']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # Population-specific categorical features (if population-aware)
        if population_aware:
            pop_categorical = ['population_name', 'pop_consequence', 'pop_gene']
            pop_categorical = [col for col in pop_categorical if col in df.columns]
            categorical_cols.extend(pop_categorical)
        
        # Sequence features to encode
        sequence_cols = ['allele', 'allele_string']
        sequence_cols = [col for col in sequence_cols if col in df.columns]
        
        self.logger.info(f"Numeric columns: {numeric_cols}")
        self.logger.info(f"Categorical columns: {categorical_cols}")
        self.logger.info(f"Sequence columns: {sequence_cols}")
        
        return numeric_cols, categorical_cols, sequence_cols
    
    def get_feature_names(self, column_transformer) -> List[str]:
        """Get output feature names from a fitted ColumnTransformer, skipping 'seq' hashed features"""
        feature_names = []
        
        # Iterate over named transformers
        for name, trans, cols in column_transformer.transformers:
            # Skip dropped columns
            if trans == 'drop':
                continue
                
            # Passthrough columns: use original column names
            if trans == 'passthrough':
                if cols is None:
                    continue
                # Get column names from indices or slice if provided
                if hasattr(column_transformer, 'feature_names_in_'):
                    # If fitted on a DataFrame, feature_names_in_ is available
                    if isinstance(cols, slice):
                        col_names = column_transformer.feature_names_in_[cols].tolist()
                    else:
                        # cols could be a list of indices or names
                        col_names = [
                            column_transformer.feature_names_in_[col] if not isinstance(col, str) else col
                            for col in (cols if isinstance(cols, (list, tuple, set)) else [cols])
                        ]
                else:
                    # Fallback: use provided column identifiers as names
                    if isinstance(cols, slice):
                        # Generate generic names x0, x1, ... if feature_names_in_ not available
                        start, stop, step = cols.start or 0, cols.stop or column_transformer._n_features, cols.step or 1
                        col_names = [f"x{i}" for i in range(start, stop, step)]
                    else:
                        col_names = list(cols) if isinstance(cols, (list, tuple, set)) else [cols]
                # Prefix with transformer name
                feature_names.extend([f"{name}__{col}" for col in col_names])
                continue
            
            try:
                # If the transformer is a Pipeline, get names from its last step
                if isinstance(trans, Pipeline):
                    # Try pipeline's own get_feature_names_out (sklearn 1.1+ may support this)
                    try:
                        names = trans.get_feature_names_out()
                    except Exception:
                        # Fall back to last step in pipeline
                        last_step = trans.steps[-1][1]
                        if hasattr(last_step, 'get_feature_names_out'):
                            # Determine input feature names for the last step
                            input_feats = None
                            if cols is not None:
                                if hasattr(column_transformer, 'feature_names_in_'):
                                    if isinstance(cols, slice):
                                        input_feats = column_transformer.feature_names_in_[cols].tolist()
                                    else:
                                        input_feats = [
                                            column_transformer.feature_names_in_[col] if not isinstance(col, str) else col
                                            for col in (cols if isinstance(cols, (list, tuple, set)) else [cols])
                                        ]
                                else:
                                    input_feats = list(cols) if isinstance(cols, (list, tuple, set)) else [cols]
                            # Get feature names from the last step (e.g., OneHotEncoder)
                            try:
                                names = last_step.get_feature_names_out(input_feats)
                            except TypeError:
                                names = last_step.get_feature_names_out()
                        else:
                            # Last step has no get_feature_names_out
                            raise AttributeError
                else:
                    # Single transformer (not a Pipeline)
                    if hasattr(trans, 'get_feature_names_out'):
                        # Use get_feature_names_out, providing input feature names if possible
                        input_features = None
                        if cols is not None:
                            if hasattr(column_transformer, 'feature_names_in_'):
                                if isinstance(cols, slice):
                                    input_features = column_transformer.feature_names_in_[cols].tolist()
                                else:
                                    input_features = [
                                        column_transformer.feature_names_in_[col] if not isinstance(col, str) else col
                                        for col in (cols if isinstance(cols, (list, tuple, set)) else [cols])
                                    ]
                            else:
                                input_features = list(cols) if isinstance(cols, (list, tuple, set)) else [cols]
                        try:
                            names = trans.get_feature_names_out(input_features)
                        except TypeError:
                            # Some transformers don't accept input_features param
                            names = trans.get_feature_names_out()
                    elif hasattr(trans, 'get_feature_names'):
                        # Support older get_feature_names (if implemented)
                        names = trans.get_feature_names(cols) if cols is not None else trans.get_feature_names()
                    else:
                        # No feature name method, raise to trigger except block
                        raise AttributeError
            except Exception:
                # If transformer does not support feature names
                if name == 'seq':
                    # Skip sequence hashed features silently
                    continue
                else:
                    # For other transformers, fallback to using the input column names
                    if cols is None:
                        names = []
                    elif hasattr(column_transformer, 'feature_names_in_'):
                        if isinstance(cols, slice):
                            names = column_transformer.feature_names_in_[cols].tolist()
                        else:
                            names = [
                                column_transformer.feature_names_in_[col] if not isinstance(col, str) else col
                                for col in (cols if isinstance(cols, (list, tuple, set)) else [cols])
                            ]
                    else:
                        # Last resort: use provided column identifiers as names
                        if isinstance(cols, slice):
                            start, stop, step = cols.start or 0, cols.stop or column_transformer._n_features, cols.step or 1
                            names = [f"x{i}" for i in range(start, stop, step)]
                        else:
                            names = list(cols) if isinstance(cols, (list, tuple, set)) else [cols]
            # Prefix extracted names with transformer name and collect
            feature_names.extend([f"{name}__{fn}" for fn in names])
            
        return feature_names
    
    def build_preprocessor(self, numeric_features, categorical_features, sequence_features, numeric_scaler='none'):
        """Build a preprocessing pipeline for the given feature sets"""
        transformers = []
        
        # Numeric
        if numeric_features:
            if numeric_scaler == 'standard':
                num_tr = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
            elif numeric_scaler == 'robust':
                num_tr = Pipeline([('imputer', SimpleImputer()), ('scaler', RobustScaler())])
            elif numeric_scaler == 'quantile':
                num_tr = Pipeline([('imputer', SimpleImputer()), 
                                ('scaler', QuantileTransformer(output_distribution='normal'))])
            else:
                num_tr = SimpleImputer()
            transformers.append(('num', num_tr, numeric_features))
        
        # Categorical
        if categorical_features:
            cat_tr = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_tr, categorical_features))
        
        # We'll skip the sequence features processing here as we'll handle them differently
        # The extract_sequence_features method already created numerical features from the sequences
        
        return ColumnTransformer(transformers=transformers, verbose_feature_names_out=True)
    
    def population_stratified_split(self, df: pd.DataFrame,
                                   test_size: float = 0.2,
                                   random_state: int = 42) -> Tuple:
        """Split data while preserving population stratification"""
        self.logger.info(f"Performing population-stratified data split")
        
        # Features and target
        X = df.drop(['clinical_significance', 'mapped_target', 'target_encoded'],
                    axis=1, errors='ignore')
        y = df['target_encoded']
        
        # Check if population column exists
        if 'population_name' in df.columns:
            # Split by population
            train_indices, test_indices = [], []
            for pop in df['population_name'].unique():
                pop_idx = df[df['population_name'] == pop].index
                if len(pop_idx) > 0:
                    # For each population, stratify by target
                    pop_y = y[pop_idx]
                    stratify = pop_y if len(pop_y.unique()) > 1 else None
                    pop_train_idx, pop_test_idx = train_test_split(
                        pop_idx, test_size=test_size,
                        random_state=random_state, stratify=stratify
                    )
                    train_indices.extend(pop_train_idx)
                    test_indices.extend(pop_test_idx)
            
            # Create train/test sets
            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y.loc[train_indices]
            y_test = y.loc[test_indices]
            
            # Store population info for fairness evaluation
            pop_train = df.loc[train_indices, 'population_name']
            pop_test = df.loc[test_indices, 'population_name']
        else:
            # Regular stratified split if no population info
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            pop_train = None
            pop_test = None
        
        self.logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test, pop_train, pop_test


class ModelManager:
    """Class for handling model creation, training, and evaluation"""
    
    def __init__(self, logger: Logger = None):
        """Initialize with optional logger"""
        self.logger = logger or Logger()
        # Dictionary to map model names to their classes
        self.model_map = {
            'RandomForest': RandomForestClassifier,
            'XGBoost': XGBClassifier,
            'LogisticRegression': LogisticRegression
        }
    
    def get_model_instance(self, name, random_state=42):
        """Create a model instance based on the name"""
        if name == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=100, 
                max_depth=10,
                class_weight='balanced', 
                random_state=random_state
            )
        elif name == 'XGBoost':
            return XGBClassifier(
                n_estimators=100, 
                max_depth=5,
                learning_rate=0.1, 
                eval_metric='mlogloss',
                random_state=random_state
            )
        elif name == 'LogisticRegression':
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model: {name}")
    
    def train_evaluate_model(self, X_train, y_train, X_test, y_test,
                             pop_test=None,
                             model_name: str = None,
                             estimator=None,
                             random_state: int = 42):
        """Train and evaluate a model with detailed metrics"""
        model_desc = model_name or (estimator.__class__.__name__ if estimator else "Unknown model")
        self.logger.info(f"Training {model_desc} model")
        
        # 1) Pick model
        if estimator is not None:
            model = estimator
        else:
            model = self.get_model_instance(model_name, random_state)
        
        # 2) Train
        model.fit(X_train, y_train)
        
        # 3) Predict
        y_pred = model.predict(X_test)
        
        # Calculate basic metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        # Print classification report
        self.logger.info(f"\nClassification Report:")
        report = classification_report(
            y_test, y_pred, target_names=['benign', 'pathogenic', 'uncertain']
        )
        self.logger.info(report)
        
        # Population fairness if available
        pop_fairness = {}
        if pop_test is not None:
            self.logger.info(f"\nAccuracy by Population Group:")
            for pop in pop_test.unique():
                pop_mask = pop_test == pop
                if sum(pop_mask) > 0:  # Only consider populations with samples
                    pop_acc = accuracy_score(y_test[pop_mask], y_pred[pop_mask])
                    self.logger.info(f"  {pop}: {pop_acc:.3f}")
                    pop_fairness[pop] = pop_acc
            
            # Calculate max accuracy difference
            max_diff = max(pop_fairness.values()) - min(pop_fairness.values())
            self.logger.info(f"Max accuracy difference: {max_diff:.3f}")
        
        # Return metrics
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model
        }
        
        if pop_test is not None and pop_fairness:
            results['fairness'] = pop_fairness
            results['max_accuracy_diff'] = max_diff
        
        return results
    
    def group_metrics_pop(self, y_true, y_pred, population_column, df):
        """Calculate metrics grouped by population"""
        results = {}
        for pop_name, group_df in df.groupby(population_column):
            group_indices = group_df.index
            group_y_true = y_true[group_indices]
            group_y_pred = y_pred[group_indices]
            
            # Calculate metrics for this population group
            accuracy = accuracy_score(group_y_true, group_y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                group_y_true, group_y_pred, average='weighted'
            )
            
            results[pop_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'count': len(group_indices)
            }
        
        return results


class Visualizer:
    """Class for creating visualizations of results"""
    
    def __init__(self, logger: Logger = None):
        """Initialize with optional logger"""
        self.logger = logger or Logger()
    
    def visualize_results(self, pop_results, nonpop_results):
        """Create comparison visualizations for the models"""
        
        # Create overall performance comparison
        plt.figure(figsize=(14, 10))
        
        # Accuracy subplot
        plt.subplot(2, 2, 1)
        models = ['Population-Aware', 'Non-Population-Aware']
        accuracies = [pop_results['accuracy'], nonpop_results['accuracy']]
        plt.bar(models, accuracies, color=['#3498db', '#e74c3c'])
        plt.ylim(max(0.7, min(accuracies) - 0.05), min(1.0, max(accuracies) + 0.05))
        plt.title('Overall Accuracy')
        
        # F1 Score subplot
        plt.subplot(2, 2, 2)
        f1_scores = [pop_results['f1'], nonpop_results['f1']]
        plt.bar(models, f1_scores, color=['#3498db', '#e74c3c'])
        plt.ylim(max(0.7, min(f1_scores) - 0.05), min(1.0, max(f1_scores) + 0.05))
        plt.title('F1 Score (Weighted)')
        
        # Fairness comparison if available
        if 'fairness' in pop_results and 'fairness' in nonpop_results:
            # Accuracy Gap subplot
            plt.subplot(2, 2, 3)
            gaps = [pop_results['max_accuracy_diff'], nonpop_results['max_accuracy_diff']]
            plt.bar(models, gaps, color=['#3498db', '#e74c3c'])
            plt.title('Max Accuracy Gap by Population (Lower is Better)')
            
            # Population-specific accuracy subplot
            plt.subplot(2, 2, 4)
            
            # Combine population data
            all_pops = sorted(set(list(pop_results['fairness'].keys()) +
                                list(nonpop_results['fairness'].keys())))
            
            # Set up bar positions
            x = np.arange(len(all_pops))
            width = 0.35
            
            # Extract accuracies for each population
            pop_accs = [pop_results['fairness'].get(pop, 0) for pop in all_pops]
            nonpop_accs = [nonpop_results['fairness'].get(pop, 0) for pop in all_pops]
            
            # Plot grouped bars
            plt.bar(x - width/2, pop_accs, width, label='Population-Aware', color='#3498db')
            plt.bar(x + width/2, nonpop_accs, width, label='Non-Population-Aware', color='#e74c3c')
            
            plt.xlabel('Population')
            plt.ylabel('Accuracy')
            plt.title('Accuracy by Population Group')
            plt.xticks(x, all_pops, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        self.logger.info("Saved model comparison visualization to model_comparison.png")
    
    def feature_importance_plot(self, model, model_name, feature_names, top_n=20):
        """Create feature importance plot for tree-based models with meaningful feature names"""
        
        if not hasattr(model, 'feature_importances_'):
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create a mapping of importance to feature name
        feature_importance = [(importance, name) for importance, name in zip(importances, feature_names)]
        
        # Sort by importance (highest to lowest)
        feature_importance.sort(reverse=True)
        
        # Plot top features
        plt.figure(figsize=(12, 10))
        n_features = min(top_n, len(importances))
        
        # Extract sorted importances and names
        top_importances = [importance for importance, _ in feature_importance[:n_features]]
        top_names = [name for _, name in feature_importance[:n_features]]
        
        plt.barh(range(n_features), top_importances)
        plt.yticks(range(n_features), top_names)
        plt.xlabel('Feature Importance')
        plt.title(f'Top {n_features} Features ({model_name})')
        plt.tight_layout()
        plt.savefig(f'{model_name.lower().replace(" ", "_")}_feature_importance.png')
        self.logger.info(f"Saved feature importance plot to {model_name.lower().replace(' ', '_')}_feature_importance.png")
        
    def plot_eda(self, df):
        """Create exploratory data analysis plots"""
        # Y distribution
        plt.figure(figsize=(6,4))
        sns.countplot(x='mapped_target', data=df,
                    order=['benign','pathogenic','uncertain'])
        plt.title("Class Distribution")
        plt.tight_layout()
        plt.savefig("EDA_class_distribution.png")
        plt.close()
        
        # Numeric feature histograms
        for col in ['allele_freq','allele_count','allele_length']:
            if col in df.columns:
                plt.figure(figsize=(6,4))
                sns.histplot(df[col].dropna(), bins=50, kde=True)
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                plt.savefig(f"EDA_dist_{col}.png")
                plt.close()
        
        # Correlation matrix
        preprocessor = DataPreprocessor(logger=self.logger)
        num_cols, _, _ = preprocessor.prepare_column_lists(df, population_aware=True)
        if num_cols:
            corr = df[num_cols].corr()
            plt.figure(figsize=(8,6))
            sns.heatmap(corr, cmap='coolwarm', center=0)
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig("EDA_corr_matrix.png")
            plt.close()
        
        self.logger.info("EDA plots saved.")
        
    def plot_cv_results(self, cv_df):
        """Plot cross-validation results"""
        plt.figure(figsize=(8,4))
        sns.barplot(data=cv_df, x='model', y='acc', hue='scaler')
        plt.title("CV Accuracy by Model & Scaler")
        plt.tight_layout()
        plt.savefig("cv_accuracy_bar.png")
        plt.close()
        self.logger.info("Saved cv_accuracy_bar.png")


class GeneticVariantAnalysis:
    """Main class for genetic variant analysis workflow"""
    
    def __init__(self, file_path: str, logger: Logger = None):
        """Initialize with file path and optional logger"""
        self.file_path = file_path
        self.logger = logger or Logger()
        self.data_loader = DataLoader(file_path, self.logger)
        self.preprocessor = DataPreprocessor(self.logger)
        self.model_manager = ModelManager(self.logger)
        self.visualizer = Visualizer(self.logger)
        
    def run_exploratory_analysis(self):
        """Run exploratory data analysis"""
        df = self.data_loader.load_data()
        df = self.preprocessor.prepare_target(df)
        df = self.preprocessor.extract_sequence_features(df)
        df_pop = self.preprocessor.create_population_features(df.copy())
        
        # Create EDA plots
        self.visualizer.plot_eda(df_pop)
        
        return df_pop
    
    def run_cross_validation(self, df):
        """Run detailed cross-validation with multiple models and scalers"""
        X = df.drop(['clinical_significance', 'mapped_target', 'target_encoded', 'allele', 'allele_string'], 
                axis=1, errors='ignore')
        y = df['target_encoded']
        
        results = []
        for scaler_name in self.preprocessor.scalers:
            num_cols, cat_cols, _ = self.preprocessor.prepare_column_lists(df, population_aware=True)
            # Remove any sequence columns from the lists
            num_cols = [col for col in num_cols if col not in ['allele', 'allele_string']]
            cat_cols = [col for col in cat_cols if col not in ['allele', 'allele_string']]
            
            pre = self.preprocessor.build_preprocessor(num_cols, cat_cols, [], numeric_scaler=scaler_name)
            group_kfold = GroupKFold(n_splits=5)
            groups = df['population_name'] if 'population_name' in df.columns else None
            
            for model_name in ['RandomForest', 'XGBoost', 'LogisticRegression']:
                pipe = Pipeline([
                    ('pre', pre), 
                    ('clf', self.model_manager.get_model_instance(model_name))
                ])
                
                cv_res = cross_validate(
                    pipe, X, y,
                    cv=group_kfold if groups is not None else 5,
                    groups=groups,
                    scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
                    n_jobs=-1,
                    error_score='raise'  # This will give more detailed error messages
                )
                
                results.append({
                    'scaler': scaler_name,
                    'model': model_name,
                    'acc': cv_res['test_accuracy'].mean(),
                    'prec': cv_res['test_precision_weighted'].mean(),
                    'rec': cv_res['test_recall_weighted'].mean(),
                    'f1': cv_res['test_f1_weighted'].mean()
                })
        
        cv_df = pd.DataFrame(results)
        cv_df.to_csv('full_cv_metrics.csv', index=False)
        self.logger.info("Saved full CV metrics → full_cv_metrics.csv")
        
        # Plot CV results
        self.visualizer.plot_cv_results(cv_df)
        
        return cv_df
    
    def run_model_comparison(self, df, model_name: str = 'RandomForest', random_state: int = 42):
        """Run complete analysis with proper sequence handling and feature name extraction"""
        self.logger.info(f"\n{'='*80}\nStarting Genetic Variant Classification Analysis\n{'='*80}")
        
        # Check for existing checkpoint
        checkpoint = self.load_checkpoint("model_comparison_final")
        if checkpoint:
            self.logger.info("Resuming from final checkpoint")
            return checkpoint['pop_results'], checkpoint['nonpop_results']
        
        # 1. Extract features from sequence data
        df = self.preprocessor.extract_sequence_features(df)
        
        # 2. Create two versions: with and without population features
        df_pop = self.preprocessor.create_population_features(df.copy())
        df_nonpop = df.copy()
        
        # 3. Identify columns for each pipeline
        pop_numeric, pop_categorical, pop_sequence = self.preprocessor.prepare_column_lists(df_pop, population_aware=True)
        nonpop_numeric, nonpop_categorical, nonpop_sequence = self.preprocessor.prepare_column_lists(df_nonpop, population_aware=False)
        
        # Check for data split checkpoint
        split_checkpoint = self.load_checkpoint("model_comparison_split")
        if split_checkpoint:
            self.logger.info("Resuming from data split checkpoint")
            X_train = split_checkpoint['X_train']
            X_test = split_checkpoint['X_test']
            y_train = split_checkpoint['y_train']
            y_test = split_checkpoint['y_test']
            pop_train = split_checkpoint['pop_train']
            pop_test = split_checkpoint['pop_test']
        else:
            # 4. Split data using population stratification
            X_train, X_test, y_train, y_test, pop_train, pop_test = self.preprocessor.population_stratified_split(
                df_pop, test_size=0.2, random_state=random_state
            )
            # Save split checkpoint
            self.save_checkpoint("model_comparison_split", {
                'X_train': X_train, 
                'X_test': X_test, 
                'y_train': y_train, 
                'y_test': y_test,
                'pop_train': pop_train, 
                'pop_test': pop_test
            })
        
        # Create train/test datasets for population-aware model
        X_train_pop = X_train.copy()
        X_test_pop = X_test.copy()
        
        # Create train/test datasets for non-population-aware model
        population_cols = [col for col in X_train.columns if 'pop_' in col or 'population' in col]
        X_train_nonpop = X_train.drop(population_cols, axis=1, errors='ignore')
        X_test_nonpop = X_test.drop(population_cols, axis=1, errors='ignore')
        
        # 5. Create preprocessors with proper sequence encoding
        pop_preprocessor = self.preprocessor.build_preprocessor(pop_numeric, pop_categorical, pop_sequence)
        nonpop_preprocessor = self.preprocessor.build_preprocessor(nonpop_numeric, nonpop_categorical, nonpop_sequence)
        
        # 6. Get feature names for feature importance plots
        try:
            pop_feature_names = self.preprocessor.get_feature_names(pop_preprocessor)
            nonpop_feature_names = self.preprocessor.get_feature_names(nonpop_preprocessor)
        except AttributeError:
            # Fix for scikit-learn API change
            self.logger.warning("Using 'transformers' instead of 'transformers_' due to API change")
            # Get feature names manually from the column transformer
            pop_feature_names = []
            for name, _, cols in pop_preprocessor.transformers:
                if isinstance(cols, list):
                    for col in cols:
                        pop_feature_names.append(f"{name}__{col}")
                        
            nonpop_feature_names = []
            for name, _, cols in nonpop_preprocessor.transformers:
                if isinstance(cols, list):
                    for col in cols:
                        nonpop_feature_names.append(f"{name}__{col}")
        
        # Check for population-aware model checkpoint
        pop_model_checkpoint = self.load_checkpoint("model_comparison_pop")
        if pop_model_checkpoint:
            self.logger.info("Resuming from population-aware model checkpoint")
            pop_results = pop_model_checkpoint['pop_results']
        else:
            # 7. Train and evaluate population-aware model via Pipeline
            self.logger.info("\n=== Population-Aware Model ===")
            pipe_pop = Pipeline([
                ('pre', pop_preprocessor),
                ('clf', self.model_manager.get_model_instance(model_name, random_state))
            ])
            pipe_pop.fit(X_train_pop, y_train)
            clf = pipe_pop.named_steps['clf']
            X_train_proc = pipe_pop.named_steps['pre'].transform(X_train_pop)
            X_test_proc = pipe_pop.named_steps['pre'].transform(X_test_pop)
            pop_results = self.model_manager.train_evaluate_model(
                X_train_proc, y_train,
                X_test_proc, y_test,
                pop_test=pop_test,
                estimator=clf,
                random_state=random_state
            )
            # Save population-aware model checkpoint
            self.save_checkpoint("model_comparison_pop", {'pop_results': pop_results})
        
        # Check for non-population-aware model checkpoint
        nonpop_model_checkpoint = self.load_checkpoint("model_comparison_nonpop")
        if nonpop_model_checkpoint:
            self.logger.info("Resuming from non-population-aware model checkpoint")
            nonpop_results = nonpop_model_checkpoint['nonpop_results']
        else:
            # 8. Train and evaluate non-population-aware model via Pipeline
            self.logger.info("\n=== Non-Population-Aware Model ===")
            pipe_non = Pipeline([
                ('pre', nonpop_preprocessor),
                ('clf', self.model_manager.get_model_instance(model_name, random_state))
            ])
            pipe_non.fit(X_train_nonpop, y_train)
            clf_non = pipe_non.named_steps['clf']
            X_train_nonproc = pipe_non.named_steps['pre'].transform(X_train_nonpop)
            X_test_nonproc = pipe_non.named_steps['pre'].transform(X_test_nonpop)
            nonpop_results = self.model_manager.train_evaluate_model(
                X_train_nonproc, y_train,
                X_test_nonproc, y_test,
                pop_test=pop_test,
                estimator=clf_non,
                random_state=random_state
            )
            # Save non-population-aware model checkpoint
            self.save_checkpoint("model_comparison_nonpop", {'nonpop_results': nonpop_results})
        
        # 9. Model comparison
        self.logger.info("\n=== Model Comparison Summary ===")
        
        if 'max_accuracy_diff' in pop_results and 'max_accuracy_diff' in nonpop_results:
            self.logger.info(f"Population-Aware Max Accuracy Gap: {pop_results['max_accuracy_diff']:.3f}")
            self.logger.info(f"Non-Population-Aware Max Accuracy Gap: {nonpop_results['max_accuracy_diff']:.3f}")
            
            # Analyze which model is more fair
            if pop_results['max_accuracy_diff'] < nonpop_results['max_accuracy_diff']:
                self.logger.info("The Population-Aware model has more consistent performance across population groups.")
            elif pop_results['max_accuracy_diff'] > nonpop_results['max_accuracy_diff']:
                self.logger.info("The Non-Population-Aware model has more consistent performance across population groups.")
            else:
                self.logger.info("Both models have similar consistency across population groups.")
        
        # 10. Visualizations
        self.visualizer.visualize_results(pop_results, nonpop_results)
        
        # 11. Feature importance for tree-based models with meaningful feature names
        if hasattr(pop_results['model'], 'feature_importances_'):
            self.visualizer.feature_importance_plot(pop_results['model'], "Population-Aware Model", pop_feature_names)
        
        if hasattr(nonpop_results['model'], 'feature_importances_'):
            self.visualizer.feature_importance_plot(nonpop_results['model'], "Non-Population-Aware Model", nonpop_feature_names)
        
        # 12. Print top features with names for easier interpretation
        if hasattr(pop_results['model'], 'feature_importances_'):
            self.logger.info("\nTop 10 features for Population-Aware Model:")
            feature_importance = [(importance, name) for importance, name in
                                zip(pop_results['model'].feature_importances_, pop_feature_names)]
            feature_importance.sort(reverse=True)
            for i, (importance, name) in enumerate(feature_importance[:10]):
                self.logger.info(f"{i+1}. {name}: {importance:.4f}")
        
        if hasattr(nonpop_results['model'], 'feature_importances_'):
            self.logger.info("\nTop 10 features for Non-Population-Aware Model:")
            feature_importance = [(importance, name) for importance, name in
                                zip(nonpop_results['model'].feature_importances_, nonpop_feature_names)]
            feature_importance.sort(reverse=True)
            for i, (importance, name) in enumerate(feature_importance[:10]):
                self.logger.info(f"{i+1}. {name}: {importance:.4f}")
        
        # Save final checkpoint
        self.save_checkpoint("model_comparison_final", {
            'pop_results': pop_results,
            'nonpop_results': nonpop_results
        })
        
        self.logger.info(f"\n{'='*80}\nAnalysis Complete\n{'='*80}")
        
        return pop_results, nonpop_results
        
        def run_hyperparameter_tuning(self, df, model_name='XGBoost'):
            """Run hyperparameter tuning for a selected model"""
            self.logger.info(f"\n=== Hyperparameter Tuning for {model_name} ===")
            
            # Prepare data
            X = df.drop(['clinical_significance', 'mapped_target', 'target_encoded'], axis=1, errors='ignore')
            y = df['target_encoded']
            
            # Create preprocessor
            num_cols, cat_cols, seq_cols = self.preprocessor.prepare_column_lists(df, population_aware=True)
            pre = self.preprocessor.build_preprocessor(num_cols, cat_cols, seq_cols, numeric_scaler='standard')
            
            # Setup base pipeline
            if model_name == 'XGBoost':
                base_model = self.model_manager.get_model_instance('XGBoost')
                pipe = Pipeline([('pre', pre), ('clf', base_model)])
                
                param_grid = {
                    'clf__n_estimators': [100, 200],
                    'clf__max_depth': [3, 5, 7],
                    'clf__learning_rate': [0.01, 0.1],
                    'clf__reg_alpha': [0, 1],
                    'clf__reg_lambda': [1, 10]
                }
            elif model_name == 'RandomForest':
                base_model = self.model_manager.get_model_instance('RandomForest')
                pipe = Pipeline([('pre', pre), ('clf', base_model)])
                
                param_grid = {
                    'clf__n_estimators': [50, 100, 200],
                    'clf__max_depth': [5, 10, 15],
                    'clf__min_samples_split': [2, 5, 10],
                    'clf__min_samples_leaf': [1, 2, 4]
                }
            elif model_name == 'LogisticRegression':
                base_model = self.model_manager.get_model_instance('LogisticRegression')
                pipe = Pipeline([('pre', pre), ('clf', base_model)])
                
                param_grid = {
                    'clf__C': [0.1, 1.0, 10.0],
                    'clf__solver': ['liblinear', 'saga'],
                    'clf__penalty': ['l1', 'l2']
                }
            else:
                raise ValueError(f"Hyperparameter tuning not configured for model: {model_name}")
            
            # Setup cross-validation
            group_kfold = GroupKFold(n_splits=5)
            groups = df['population_name'] if 'population_name' in df.columns else None
            
            # Run randomized search
            search = RandomizedSearchCV(
                pipe, param_distributions=param_grid,
                n_iter=20,
                cv=group_kfold if groups is not None else 5,
                groups=groups,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2,
                random_state=42
            )
            
            # Split data for final evaluation
            X_train, X_test, y_train, y_test, pop_train, pop_test = self.preprocessor.population_stratified_split(
                df, test_size=0.2, random_state=42
            )
            
            # Fit the search
            search.fit(X_train, y_train)
            
            # Log results
            self.logger.info(f"Best parameters: {search.best_params_}")
            self.logger.info(f"Best CV accuracy: {search.best_score_:.4f}")
            
            # Evaluate on test set
            best_pred = search.predict(X_test)
            self.logger.info("Test set performance:")
            report = classification_report(
                y_test, best_pred, target_names=['benign', 'pathogenic', 'uncertain']
            )
            self.logger.info(report)
            
            return search.best_estimator_, search.best_params_, search.best_score_
    
    def run_holdout_evaluation(self, df, best_models=None):
        """Run hold-out evaluation comparing population-aware vs non-population-aware models"""
        self.logger.info("\n=== Hold-out Evaluation ===")
        
        # Split once to get both pop and non-pop datasets
        X_train, X_test, y_train, y_test, pop_train, pop_test = self.preprocessor.population_stratified_split(
            df, test_size=0.2, random_state=42
        )
        
        # Create non-pop copies
        population_cols = [c for c in X_train.columns if 'pop_' in c or c == 'population_name']
        X_train_nonpop = X_train.drop(population_cols, axis=1, errors='ignore')
        X_test_nonpop = X_test.drop(population_cols, axis=1, errors='ignore')
        
        results = {}
        
        # If best_models is provided, use those; otherwise use default models
        if best_models is None:
            best_models = {
                'population_aware': {
                    'scaler': 'standard',
                    'model': 'RandomForest'
                },
                'nonpopulation_aware': {
                    'scaler': 'standard',
                    'model': 'RandomForest'
                }
            }
        
        for mode, (X_tr, X_te, pop_te) in [
            ('population_aware', (X_train, X_test, pop_test)),
            ('nonpopulation_aware', (X_train_nonpop, X_test_nonpop, pop_test))
        ]:
            self.logger.info(f"\n=== Hold-out eval: {mode} ===")
            
            # Get model configuration
            scaler_name = best_models[mode]['scaler']
            model_name = best_models[mode]['model']
            
            # Build the appropriate pipeline
            num_cols, cat_cols, seq_cols = self.preprocessor.prepare_column_lists(
                df, population_aware=(mode == 'population_aware')
            )
            
            pre = self.preprocessor.build_preprocessor(
                num_cols, cat_cols, seq_cols,
                numeric_scaler=scaler_name
            )
            
            est = self.model_manager.get_model_instance(model_name)
            fair_est = GridSearch(
                estimator=est,
                constraints=DemographicParity(),
                grid_size=20
            )
            
            pipe = Pipeline([('pre', pre), ('clf', fair_est)])
            
            # Optional population weighting for the aware arm
            if mode == 'population_aware' and pop_train is not None:
                weights = pop_train.map(pop_train.value_counts().rdiv(1.0))
                pipe.fit(X_tr, y_train, clf__sample_weight=weights.loc[X_tr.index])
            else:
                pipe.fit(X_tr, y_train)
            
            # Predict and report
            y_pred = pipe.predict(X_te)
            self.logger.info(f"\n>> {model_name} + {scaler_name}")
            report = classification_report(
                y_test, y_pred,
                target_names=['benign', 'pathogenic', 'uncertain']
            )
            self.logger.info(report)
            
            # Per-population accuracy
            if pop_te is not None:
                acc_by_pop = {
                    pop: (y_test[pop_te == pop] == y_pred[pop_te == pop]).mean()
                    for pop in pop_te.unique()
                }
                self.logger.info(f"Per-population accuracy: {acc_by_pop}")
            
            # Store results
            results[mode] = {
                'pipeline': pipe,
                'y_pred': y_pred,
                'accuracy': accuracy_score(y_test, y_pred),
                'population_accuracy': acc_by_pop if pop_te is not None else None
            }
        
        return results
    def save_checkpoint(self, stage: str, data: dict):
        """Save a checkpoint at a specific stage of the analysis"""
        checkpoint_dir = "checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = f"{checkpoint_dir}/{stage}_{timestamp}.pkl"
        joblib.dump(data, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Also save a pointer to the latest checkpoint for this stage
        latest_path = f"{checkpoint_dir}/{stage}_latest.txt"
        with open(latest_path, 'w') as f:
            f.write(checkpoint_path)
        
    def load_checkpoint(self, stage: str):
        """Load the latest checkpoint for a given stage if it exists"""
        latest_path = f"checkpoints/{stage}_latest.txt"
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                checkpoint_path = f.read().strip()
            
            if os.path.exists(checkpoint_path):
                self.logger.info(f"Loading checkpoint: {checkpoint_path}")
                return joblib.load(checkpoint_path)
        
        self.logger.info(f"No checkpoint found for stage: {stage}")
        return None


def main():
    """Main execution function with error handling and checkpointing"""
    # Set file path
    FILE_PATH = "raw_variant_data.csv"
    
    # Create logger
    logger = Logger()
    
    try:
        # Create analysis object
        analysis = GeneticVariantAnalysis(FILE_PATH, logger)
        
        # Step 1: Run EDA
        logger.info("\n=== Running Exploratory Data Analysis ===")
        try:
            df_pop = analysis.run_exploratory_analysis()
            
            # Step 2: Run cross-validation
            logger.info("\n=== Running Cross-Validation ===")
            try:
                cv_results = analysis.run_cross_validation(df_pop)
                
                # Step 3: Run model comparison
                logger.info("\n=== Running Model Comparison ===")
                try:
                    pop_results, nonpop_results = analysis.run_model_comparison(df_pop, model_name='RandomForest')
                    
                    # Step 4: Run hyperparameter tuning
                    logger.info("\n=== Running Hyperparameter Tuning ===")
                    try:
                        best_model, best_params, best_score = analysis.run_hyperparameter_tuning(df_pop, model_name='XGBoost')
                        
                        # Step 5: Run hold-out evaluation
                        logger.info("\n=== Running Hold-out Evaluation ===")
                        try:
                            holdout_results = analysis.run_holdout_evaluation(df_pop)
                        except Exception as e:
                            logger.error(f"Error in hold-out evaluation: {str(e)}")
                            import traceback
                            logger.error(traceback.format_exc())
                    except Exception as e:
                        logger.error(f"Error in hyperparameter tuning: {str(e)}")
                        import traceback
                        logger.error(traceback.format_exc())
                except Exception as e:
                    logger.error(f"Error in model comparison: {str(e)}")
                    import traceback
                    logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"Error in cross-validation: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error in exploratory analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()