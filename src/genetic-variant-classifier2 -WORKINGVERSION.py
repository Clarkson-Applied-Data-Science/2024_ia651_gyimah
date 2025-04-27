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
from tqdm import tqdm                     
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
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    average_precision_score, precision_recall_curve, auc
)
from post_training_utils import plot_confusion
# --------------------------------------------------------------------
# Silence noisy, non-fatal warnings that clutter the console
# --------------------------------------------------------------------
import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=FutureWarning)        
warnings.filterwarnings("ignore", category=ConvergenceWarning)

class Logger:
    """
    Logging utility class for consistent formatting and easy logging.
    """
    def __init__(self, level=logging.INFO, name: str = __name__):
        """
        Initialize the logger with a specified log level and name.
        """
        # Configure the basic logger only once for the given logger name
        logger = logging.getLogger(name)
        if not logger.handlers:
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(levelname)s - %(message)s"
            )
        self.logger = logger
        self.logger.setLevel(level)
    
    def info(self, message: str) -> None:
        """Log an info-level message."""
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """Log a warning-level message."""
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """Log an error-level message."""
        self.logger.error(message)


class DataLoader:
    """
    Class for loading and initialising processing of genetic variant data.
    """
    def __init__(self, file_path: str, logger: Logger = None):
        """
        Initialize with file path and optional logger.
        """
        self.file_path = file_path
        self.logger = logger or Logger()
    
    def load_data(self) -> pd.DataFrame:
        """
        Load genetic variant data with appropriate dtypes and safe numeric conversion.
        """
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
        
        # Read the data with error handling
        try:
            df = pd.read_csv(self.file_path, dtype=dtype_mapping, low_memory=False)
        except Exception as e:
            self.logger.error(f"Failed to load data: {e}")
            raise
        
        # Convert numeric columns safely
        numeric_cols = ['allele_freq', 'allele_count', 'sample_size', 'position']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        self.logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        return df



class DataPreprocessor:
    """
    Class for data preprocessing and feature engineering.
    """
    def __init__(self, logger: Logger = None):
        self.logger = logger or Logger()
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
        sequence columns. Return a 1-D array-like of concatenated strings,
        one per sample.
        """
        if isinstance(X, pd.DataFrame):
            return X.astype(str).apply(''.join, axis=1)
        return np.array([''.join(map(str, row)) for row in X])
    
    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare the target variable (clinical significance), mapping to categories.
        """
        self.logger.info("Preparing target variable")
        df = df.copy()
        
        def map_to_classes(label_str):
            if pd.isna(label_str):
                return np.nan
            label = str(label_str).lower()
            if "pathogenic" in label:
                return "pathogenic"
            if "benign" in label and "pathogenic" not in label:
                return "benign"
            if "uncertain" in label or "conflicting" in label:
                return "uncertain"
            return np.nan
        
        df["mapped_target"] = df["clinical_significance"].apply(map_to_classes)
        before_drop = df.shape[0]
        df = df.dropna(subset=["mapped_target"])
        dropped = before_drop - df.shape[0]
        if dropped:
            self.logger.warning(f"Dropped {dropped} rows with undefined target classes")
        
        label_map = {"benign": 0, "pathogenic": 1, "uncertain": 2}
        df["target_encoded"] = df["mapped_target"].map(label_map)
        
        class_dist = df["mapped_target"].value_counts()
        self.logger.info(f"Target class distribution:\n{class_dist}")
        
        # (Optional) save or plot distribution
        plt.figure(figsize=(8,6))
        class_dist.plot(kind='bar')
        plt.title('Class Distribution')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.savefig("class_distribution.png")
        
        return df
    
    def extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract numerical features from sequence data (e.g., allele, allele_string).
        """
        self.logger.info("Extracting sequence-based features")
        df = df.copy()
        
        # Process 'allele' column if present
        if 'allele' in df.columns:
            df['allele'] = df['allele'].fillna('').astype(str)
            df['allele_length'] = df['allele'].str.len()
            df['allele_A_count'] = df['allele'].str.count('A')
            df['allele_C_count'] = df['allele'].str.count('C')
            df['allele_G_count'] = df['allele'].str.count('G')
            df['allele_T_count'] = df['allele'].str.count('T')
            # Avoid division by zero
            df['allele_GC_content'] = (df['allele_G_count'] + df['allele_C_count']) / df['allele_length'].replace(0, 1)
        
        # If 'allele' not present, try 'allele_string'
        elif 'allele_string' in df.columns:
            df['allele_string'] = df['allele_string'].fillna('').astype(str)
            df['allele_length'] = df['allele_string'].str.len()
            df['allele_A_count'] = df['allele_string'].str.count('A')
            df['allele_C_count'] = df['allele_string'].str.count('C')
            df['allele_G_count'] = df['allele_string'].str.count('G')
            df['allele_T_count'] = df['allele_string'].str.count('T')
            df['allele_GC_content'] = (df['allele_G_count'] + df['allele_C_count']) / df['allele_length'].replace(0, 1)
        else:
            self.logger.info("No allele or allele_string column found for sequence features")
            # If neither column exists, return without adding features
            return df
        
        seq_cols = ['allele_length', 'allele_A_count', 'allele_C_count', 'allele_G_count', 'allele_T_count', 'allele_GC_content']
        seq_cols = [col for col in seq_cols if col in df.columns]
        self.logger.info(f"Extracted sequence features: {seq_cols}")
        return df

    def create_population_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create population-specific interaction features.
        """
        self.logger.info("Creating population-aware features")
        df = df.copy()
        
        if 'population_name' in df.columns:
            # pop_consequence: concatenation of population and consequence type
            if 'consequence_type' in df.columns:
                df['pop_consequence'] = df['population_name'].astype(str) + '_' + df['consequence_type'].astype(str)
            # pop_gene: concatenation of population and gene
            if 'gene' in df.columns:
                df['pop_gene'] = df['population_name'].astype(str) + '_' + df['gene'].astype(str)
            # allele frequency relative to population mean
            if 'allele_freq' in df.columns:
                pop_freq_means = df.groupby(df['population_name'].astype(str))['allele_freq'].mean().to_dict()
                df['pop_allele_freq_mean'] = df['population_name'].astype(str).map(pop_freq_means)
                df['pop_allele_freq_mean'] = df['pop_allele_freq_mean'].fillna(df['allele_freq'].mean())
                df['allele_freq_rel'] = df['allele_freq'] / df['pop_allele_freq_mean'].replace(0, np.nan)
        
        return df

    def prepare_column_lists(self, df: pd.DataFrame, population_aware: bool = True) -> Tuple[list, list, list]:
        """
        Identify numeric, categorical, and sequence columns for preprocessing.
        """
        # Basic numeric features
        numeric_cols = [
            'allele_freq', 'allele_count', 'sample_size', 'position',
            'allele_length', 'allele_A_count', 'allele_C_count',
            'allele_G_count', 'allele_T_count', 'allele_GC_content'
        ]
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        # Include relative allele frequency if population-aware
        if population_aware and 'allele_freq_rel' in df.columns:
            numeric_cols.append('allele_freq_rel')
        
        # Basic categorical features
        categorical_cols = ['gene', 'chromosome', 'consequence_type', 'region']
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        # Population-specific categorical features
        if population_aware:
            pop_cats = ['population_name', 'pop_consequence', 'pop_gene']
            pop_cats = [col for col in pop_cats if col in df.columns]
            categorical_cols.extend(pop_cats)
        
        # Sequence columns (to encode if we had any sequence-based encoders; currently we treat them via features)
        sequence_cols = ['allele', 'allele_string']
        sequence_cols = [col for col in sequence_cols if col in df.columns]
        
        self.logger.info(f"Numeric columns: {numeric_cols}")
        self.logger.info(f"Categorical columns: {categorical_cols}")
        self.logger.info(f"Sequence columns: {sequence_cols}")
        return numeric_cols, categorical_cols, sequence_cols

    def get_feature_names(self, column_transformer) -> list:
        """
        Get output feature names from a fitted ColumnTransformer.
        """
        try:
            # Assumes transformer is already fitted (so feature_names_in_ is available)
            return list(column_transformer.get_feature_names_out())
        except Exception as e:
            self.logger.error("Unable to retrieve feature names: ensure the ColumnTransformer is fitted. ")
            return []
    
    def build_preprocessor(self, numeric_features, categorical_features, sequence_features, numeric_scaler='none'):
        """Build a preprocessing pipeline for the given feature sets"""
        transformers = []

        # Numeric pipeline
        if numeric_features:
            if numeric_scaler == 'standard':
                num_tr = Pipeline([('imputer', SimpleImputer()), ('scaler', StandardScaler())])
            elif numeric_scaler == 'robust':
                num_tr = Pipeline([('imputer', SimpleImputer()), ('scaler', RobustScaler())])
            elif numeric_scaler == 'quantile':
                num_tr = Pipeline([('imputer', SimpleImputer()), ('scaler', QuantileTransformer(output_distribution='normal'))])
            else:
                num_tr = SimpleImputer()
            transformers.append(('num', num_tr, numeric_features))

        # Categorical pipeline
        if categorical_features:
            cat_tr = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_tr, categorical_features))

        # Sequence pipeline: concatenate sequences then vectorize
        if sequence_features:
            seq_tr = Pipeline([
                ('concat', FunctionTransformer(DataPreprocessor._concat_sequences, validate=False)),
                ('vectorizer', CountVectorizer(analyzer='char', token_pattern=None))
            ])
            transformers.append(('seq', seq_tr, sequence_features))

        return ColumnTransformer(transformers=transformers, verbose_feature_names_out=True)
    
    def population_stratified_split(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """
        Split data while preserving population stratification (stratify within each population).
        Returns X_train, X_test, y_train, y_test, pop_train, pop_test.
        """
        self.logger.info("Performing population-stratified data split")
        
        X = df.drop(['clinical_significance', 'mapped_target', 'target_encoded'], axis=1, errors='ignore')
        y = df['target_encoded']
        
        if 'population_name' in df.columns:
            train_indices, test_indices = [], []
            for pop in df['population_name'].unique():
                pop_idx = df[df['population_name'] == pop].index
                if len(pop_idx) < 2:
                    self.logger.warning(f"Population '{pop}' has less than 2 samples; assigning all to training set.")
                    train_indices.extend(pop_idx.tolist())
                    continue
                pop_y = y.loc[pop_idx]
                stratify = pop_y if pop_y.nunique() > 1 else None
                tr_idx, te_idx = train_test_split(
                    pop_idx,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify
                )
                train_indices.extend(tr_idx.tolist())
                test_indices.extend(te_idx.tolist())
            
            # Create train/test sets
            X_train = X.loc[train_indices].reset_index(drop=True)
            X_test = X.loc[test_indices].reset_index(drop=True)
            y_train = y.loc[train_indices].reset_index(drop=True)
            y_test = y.loc[test_indices].reset_index(drop=True)
            pop_train = df.loc[train_indices, 'population_name'].reset_index(drop=True)
            pop_test = df.loc[test_indices, 'population_name'].reset_index(drop=True)
        else:
            # No population info: regular stratified split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            pop_train = None
            pop_test = None
        
        self.logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        return X_train, X_test, y_train, y_test, pop_train, pop_test

class ACMGPopulationIntegration:
    """
    Integrates ACMG/AMP guidelines with population-specific information
    for variant classification.
    """
    def __init__(self, population_weights=None):
        """
        Initialize ACMG criteria weights and optional population adjustments.
        """
        # ACMG criteria base weights
        self.acmg_criteria = {
            'PVS1': 8, 'PS1': 4, 'PS2': 4, 'PS3': 4, 'PS4': 4,
            'PM1': 2, 'PM2': 2, 'PM3': 2, 'PM4': 2, 'PM5': 2, 'PM6': 2,
            'PP1': 1, 'PP2': 1, 'PP3': 1, 'PP4': 1, 'PP5': 1,
            'BA1': -8, 'BS1': -4, 'BS2': -4, 'BS3': -4, 'BS4': -4,
            'BP1': -1, 'BP2': -1, 'BP3': -1, 'BP4': -1,
            'BP5': -1, 'BP6': -1, 'BP7': -1
        }
        
        # Population-specific adjustment factors (multipliers)
        self.population_weights = population_weights or {
            'African American': {'PM2': 1.2, 'PP1': 1.1},
            'African Caribbean': {'PM2': 1.2, 'PP1': 1.1},
            'Central European': {'PM2': 0.9, 'PP1': 1.0},
            'Han Chinese': {'PM2': 1.1, 'PP1': 0.9},
            'Esan': {'PM2': 1.2, 'PP1': 1.1},
            'Gambian': {'PM2': 1.2, 'PP1': 1.1},
            'Japanese': {'PM2': 1.1, 'PP1': 0.9},
            'Luhya': {'PM2': 1.2, 'PP1': 1.1},
            'Mende': {'PM2': 1.2, 'PP1': 1.1},
            'Yoruba': {'PM2': 1.2, 'PP1': 1.1},
        }
    
    def calculate_score(self, criteria_list, population=None) -> float:
        """
        Calculate ACMG-based score with optional population adjustment.
        """
        base_score = 0.0
        # Sum base weights for each criterion
        for criterion in criteria_list:
            base_score += self.acmg_criteria.get(criterion, 0)
        # Apply population-specific multipliers
        if population in self.population_weights:
            adjustments = self.population_weights[population]
            for criterion in criteria_list:
                if criterion in adjustments:
                    base_score *= adjustments[criterion]
        return base_score
    
    def integrate_with_ml_model(self, ml_predictions, criteria_lists, populations):
        """
        Integrate ML model predictions (probabilities) with ACMG-based scores.
        
        Args:
            ml_predictions: List or array of ML model probabilities (for pathogenic class).
            criteria_lists: List of lists of ACMG criteria per variant.
            populations: List of population identifiers per variant.
        
        Returns:
            integrated_predictions: List of combined scores (0-1 scale).
        """
        integrated_predictions = []
        # Define normalization range (example)
        max_score = 20
        min_score = -20
        alpha = 0.7  # Weight for ML predictions
        
        for ml_pred, criteria, pop in zip(ml_predictions, criteria_lists, populations):
            acmg_score = self.calculate_score(criteria, pop)
            # Normalize ACMG score to [0,1]
            normalized_acmg = (acmg_score - min_score) / (max_score - min_score)
            # Combined score
            combined = alpha * ml_pred + (1 - alpha) * normalized_acmg
            integrated_predictions.append(combined)
        
        return integrated_predictions





class FairnessAwareClassifier:
    """
    A wrapper for any sklearn classifier that enforces fairness constraints
    using the equalized odds approach, now extended to multi-class classification.
    """
    def __init__(self, base_classifier, sensitive_feature_idx, random_state=42):
        """
        Initialize the fairness-aware classifier.
        
        Args:
            base_classifier: Any sklearn classifier with fit and predict methods.
            sensitive_feature_idx: Index of the sensitive feature (population) in X.
            random_state: Random seed for reproducibility.
        """
        self.base_classifier = base_classifier
        self.sensitive_feature_idx = sensitive_feature_idx
        self.random_state = random_state
        self.thresholds = {}
        self.fitted = False

    def fit(self, X, y):
        """Fit the base classifier and calculate group- and class-specific thresholds."""
        # Train the base classifier
        self.base_classifier.fit(X, y)
        self.fitted = True
        
        # Get protected attribute values
        sensitive_values = X[:, self.sensitive_feature_idx]
        unique_groups = np.unique(sensitive_values)
        
        # Get predicted probabilities for all classes
        y_prob = self.base_classifier.predict_proba(X)  # shape (n_samples, n_classes)
        classes = self.base_classifier.classes_
        n_classes = len(classes)
        
        # Initialize thresholds for each group and class (default 0.5)
        self.thresholds = {group: np.array([0.5] * n_classes) for group in unique_groups}
        
        # Target false positive rate for threshold tuning
        target_fpr = 0.1
        
        # For each class (one-vs-rest), compute thresholds per group
        for i, cls in enumerate(classes):
            # Create binary labels for current class
            y_true_class = (y == cls).astype(int)
            
            for group in unique_groups:
                group_mask = (sensitive_values == group)
                if np.sum(group_mask) == 0:
                    continue
                y_true_group = y_true_class[group_mask]
                y_prob_group = y_prob[group_mask, i]
                
                # Skip if group has only one class present
                if len(np.unique(y_true_group)) < 2:
                    continue
                
                # Compute ROC curve for this class, group
                fpr, tpr, thresh = roc_curve(y_true_group, y_prob_group)
                # Find threshold that gives FPR closest to target
                idx = np.argmin(np.abs(fpr - target_fpr))
                self.thresholds[group][i] = thresh[idx]
        
        return self

    def predict(self, X):
        """
        Return class predictions using the group-specific thresholds.
        Output is a 1-D NumPy array whose dtype matches `classes_`,
        so scikit-learn sees it as a proper multiclass target.
        """
        if not self.fitted:
            raise ValueError("Classifier not fitted. Call fit first.")

        y_prob = self.base_classifier.predict_proba(X)
        sensitive_values = X[:, self.sensitive_feature_idx]
        classes = self.base_classifier.classes_
        n_classes = len(classes)

        # allocate with the *same* dtype as the class labels (normally int64)
        y_pred = np.empty(len(X), dtype=classes.dtype)
        default_thr = np.array([0.5] * n_classes)

        for i in range(len(X)):
            group = sensitive_values[i]
            thr = self.thresholds.get(group, default_thr)
            if thr is None or len(thr) != n_classes or np.any(np.isnan(thr)):
                thr = default_thr

            scores = y_prob[i] - thr
            y_pred[i] = classes[np.argmax(scores)]

        return y_pred


    def predict_proba(self, X):
        """Return predicted probabilities from base classifier."""
        if not self.fitted:
            raise ValueError("Classifier not fitted. Call fit first.")
        return self.base_classifier.predict_proba(X)


class PathwayFeatureExtractor:
    """
    Extract pathway-level features from gene annotations to improve generalization.
    """
    def __init__(self, gene_to_pathway_map=None, pathway_importance=None):
        """
        Initialize with a gene-to-pathway mapping and optional pathway importance weights.
        """
        # Example default mapping (can be replaced by real data)
        self.gene_to_pathway_map = gene_to_pathway_map or {
            'G6PD': ['Pentose phosphate pathway', 'Metabolic pathways'],
            'BRCA1': ['DNA repair', 'Cell cycle'],
            'BRCA2': ['DNA repair', 'Cancer pathways'],
        }
        self.pathway_importance = pathway_importance or {}
        
        # Derive unique pathways and an index mapping
        self.all_pathways = sorted({
            pathway
            for paths in self.gene_to_pathway_map.values()
            for pathway in paths
        })
        self.pathway_to_idx = {pathway: i for i, pathway in enumerate(self.all_pathways)}
    
    def transform(
        self,
        X: pd.DataFrame,
        gene_column: str = "gene",
        consequence_column: str = "consequence_type"
):
        """
        Convert gene/consequence info into pathway-level numeric features and
        **append** them to the original frame.  We *do not* drop any columns so
        that downstream preprocessors can still access them.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("PathwayFeatureExtractor expects a pandas DataFrame.")

        genes = X[gene_column].astype(str).values
        consequences = X[consequence_column].astype(str).values

        n_samples = len(genes)
        n_pathways = len(self.all_pathways)
        M = np.zeros((n_samples, n_pathways))

        for i, (g, c) in enumerate(zip(genes, consequences)):
            paths = self.gene_to_pathway_map.get(g, [])
            sev = 0.5
            if "frameshift" in c:
                sev = 3.0
            elif "missense" in c:
                sev = 2.0
            for p in paths:
                idx = self.pathway_to_idx[p]
                imp = self.pathway_importance.get(p, 1.0)
                M[i, idx] = sev * imp

        pathway_df = pd.DataFrame(
            M, columns=[f"pathway_{p}" for p in self.all_pathways], index=X.index
        )
        # concatenate without dropping anything
        return pd.concat([X, pathway_df], axis=1)
    
    def fit_transform(self, X, y=None, gene_column='gene', consequence_column='consequence_type'):
        """
        Fit-transform for pipeline compatibility (no fitting needed).
        """
        return self.transform(X, gene_column, consequence_column)




class ModelManager:
    """Class for handling model creation, training, and evaluation"""
    
    def __init__(self, logger: Logger = None):
        self.logger = logger or Logger()
        # Include FairnessAware as an option
        self.model_map = {
            'RandomForest': RandomForestClassifier,
            'XGBoost': XGBClassifier,
            'LogisticRegression': LogisticRegression,
            'FairnessAware': FairnessAwareClassifier
        }
        
    def get_model_instance(self, name, random_state=42):
        """Create a model instance based on the name"""
        if name == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight='balanced', random_state=random_state
            )
        elif name == 'XGBoost':
            return XGBClassifier(
                n_estimators=100, max_depth=5,
                learning_rate=0.1, eval_metric='mlogloss',
                random_state=random_state
            )
        elif name == 'LogisticRegression':
            return LogisticRegression(
                max_iter=1000, class_weight='balanced',
                random_state=random_state
            )
        elif name == 'FairnessAware':
            # Wrap a base classifier (e.g., RandomForest) with fairness constraints
            base = RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight='balanced', random_state=random_state
            )
            # Assume the first feature is the sensitive attribute by default
            return FairnessAwareClassifier(base_classifier=base, sensitive_feature_idx=0, random_state=random_state)
        else:
            raise ValueError(f"Unknown model: {name}")

    def train_evaluate_model(self, X_train, y_train, X_test, y_test, pop_test=None, model_name: str = None, estimator=None, random_state: int = 42):
        """Train and evaluate a model with detailed metrics"""
        model_desc = model_name or (estimator.__class__.__name__ if estimator else "Unknown model")
        self.logger.info(f"Training {model_desc} model")
        
        model = estimator if estimator is not None else self.get_model_instance(model_name, random_state)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        awareness = "(Pop-Aware)" if pop_test is not None else "(Non-Pop)"
        self.logger.info("\nClassification Report {awareness}:")
        report = classification_report(y_test, y_pred, target_names=['benign', 'pathogenic', 'uncertain'])
        self.logger.info(report)
        
        results = {
            'arm': awareness,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'model': model
        }
        
        if pop_test is not None:
            df_test = pd.DataFrame({
                'population_name': pop_test.values,
                'true': y_test.values,
                'pred': y_pred
            }, index=pop_test.index)
            group_metrics = self.group_metrics_pop(df_test['true'], df_test['pred'], 'population_name', df_test)
            for pop_name, metrics in group_metrics.items():
                self.logger.info(
                    f"Metrics for {pop_name}: Acc={metrics['accuracy']:.3f}, "
                    f"Precision={metrics['precision']:.3f}, "
                    f"Recall={metrics['recall']:.3f}, F1={metrics['f1']:.3f}"
                )
            
            accuracies = [m['accuracy'] for m in group_metrics.values()]
            if accuracies:
                max_diff = max(accuracies) - min(accuracies)
                self.logger.info(f"Max accuracy difference: {max_diff:.3f}")
                results['max_accuracy_diff'] = max_diff
            
            results['group_metrics'] = group_metrics
            # Add fairness (accuracy by population) for visualization
            results['fairness'] = {pop: metrics['accuracy'] for pop, metrics in group_metrics.items()}
        
        return results


    def group_metrics_pop(self, y_true, y_pred, population_column, df):
        """Calculate metrics grouped by population"""
        results = {}
        for pop_name, group_df in df.groupby(population_column):
            group_indices = group_df.index
            # Use array or series accordingly
            if isinstance(y_true, pd.Series):
                group_y_true = y_true.loc[group_indices]
            else:
                group_y_true = y_true[group_indices]
            group_y_pred = y_pred[group_indices]
            
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
    """
    Class for creating visualizations of results.
    """
    def __init__(self, logger: Logger = None):
        self.logger = logger or Logger()
    
    def visualize_results(self, pop_results, nonpop_results):
        """
        Create comparison plots for overall accuracy, F1, and fairness metrics.
        """
        plt.figure(figsize=(14, 10))
        
        # Overall Accuracy
        plt.subplot(2, 2, 1)
        models = ['Population-Aware', 'Non-Population-Aware']
        accuracies = [pop_results['accuracy'], nonpop_results['accuracy']]
        plt.bar(models, accuracies, color=['#3498db', '#e74c3c'])
        plt.ylim(0, 1)
        plt.title('Overall Accuracy')
        
        # F1 Score (Weighted)
        plt.subplot(2, 2, 2)
        f1_scores = [pop_results['f1'], nonpop_results['f1']]
        plt.bar(models, f1_scores, color=['#3498db', '#e74c3c'])
        plt.ylim(0, 1)
        plt.title('Weighted F1 Score')
        
        # Fairness: Max Accuracy Gap by Population
        if 'max_accuracy_diff' in pop_results and 'max_accuracy_diff' in nonpop_results:
            plt.subplot(2, 2, 3)
            gaps = [pop_results['max_accuracy_diff'], nonpop_results['max_accuracy_diff']]
            plt.bar(models, gaps, color=['#3498db', '#e74c3c'])
            plt.title('Max Accuracy Gap by Population (Lower is Better)')
        
        # Population-specific accuracy bars
        if 'fairness' in pop_results or 'fairness' in nonpop_results:
            all_pops = sorted(set(pop_results.get('fairness', {}).keys()) | set(nonpop_results.get('fairness', {}).keys()))
            if all_pops:
                plt.subplot(2, 2, 4)
                x = range(len(all_pops))
                width = 0.35
                pop_accs = [pop_results.get('fairness', {}).get(p, 0.0) for p in all_pops]
                non_accs = [nonpop_results.get('fairness', {}).get(p, 0.0) for p in all_pops]
                plt.bar([xi - width/2 for xi in x], pop_accs, width, label='Pop-Aware', color='#3498db')
                plt.bar([xi + width/2 for xi in x], non_accs, width, label='Non-Pop-Aware', color='#e74c3c')
                plt.ylabel('Accuracy')
                plt.title('Accuracy by Population Group')
                plt.xticks(x, all_pops, rotation=45, ha='right')
                plt.legend()
        
        plt.tight_layout()
        plt.savefig('model_comparison.png')
        plt.close()
        self.logger.info("Saved model comparison visualization to 'model_comparison.png'")
    
    # in class Visualizer
    def feature_importance_plot(self, model, model_name, feature_names, top_n=20):
        """Safely plot importances if list lengths match."""
        if not hasattr(model, "feature_importances_"):
            self.logger.info(f"{model_name} has no feature_importances_; skipping.")
            return

        importances = model.feature_importances_
        if len(importances) != len(feature_names) or len(feature_names) == 0:
            self.logger.warning(f"Skip importance plot for {model_name} "
                                "(feature name list empty or length mismatch).")
            return

        top_n  = min(top_n, len(importances))
        idx    = np.argsort(importances)[-top_n:]

        plt.figure(figsize=(8, 6))
        plt.barh(range(top_n), importances[idx])
        plt.yticks(range(top_n), [feature_names[i] for i in idx])
        plt.xlabel("Importance")
        plt.title(f"Top {top_n} Features – {model_name}")
        plt.tight_layout()
        fname = f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
        plt.savefig(fname)
        plt.close()
        self.logger.info(f"Saved feature importance plot → {fname}")


    
    def plot_eda(self, df):
        """
        Create exploratory data analysis plots (class distribution, numeric histograms, etc.).
        """
        # Class distribution
        plt.figure(figsize=(6, 4))
        sns.countplot(x='mapped_target', data=df,
                      order=['benign','pathogenic','uncertain'])
        plt.title("Class Distribution")
        plt.tight_layout()
        plt.savefig("EDA_class_distribution.png")
        plt.close()
        
        # Histograms for numeric features
        for col in ['allele_freq', 'allele_count', 'allele_length']:
            if col in df.columns:
                plt.figure(figsize=(6, 4))
                sns.histplot(df[col].dropna(), bins=50, kde=True)
                plt.title(f"Distribution of {col}")
                plt.tight_layout()
                plt.savefig(f"EDA_dist_{col}.png")
                plt.close()
        
        # Correlation matrix of numeric features
        num_cols, _, _ = DataPreprocessor().prepare_column_lists(df, population_aware=True)
        if num_cols:
            corr = df[num_cols].corr()
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr, cmap='coolwarm', center=0, annot=True, fmt=".2f")
            plt.title("Correlation Matrix")
            plt.tight_layout()
            plt.savefig("EDA_corr_matrix.png")
            plt.close()
        
        self.logger.info("EDA plots saved.")
    
    def plot_cv_results(self, cv_df):
        """
        Plot cross-validation results (e.g., accuracy by model and scaler).
        """
        plt.figure(figsize=(8, 4))
        sns.barplot(data=cv_df, x='model', y='acc', hue='scaler')
        plt.title("CV Accuracy by Model & Scaler")
        plt.tight_layout()
        plt.savefig("cv_accuracy_bar.png")
        plt.close()
        self.logger.info("Saved 'cv_accuracy_bar.png'")





class GeneticVariantAnalysis:
    """
    Main class for orchestrating the genetic variant analysis workflow.
    """
    def __init__(self, file_path: str, logger: Logger = None):
        self.file_path = file_path
        self.logger = logger or Logger()
        self.data_loader = DataLoader(file_path, self.logger)
        self.preprocessor = DataPreprocessor(self.logger)
        self.model_manager = ModelManager(self.logger)
        self.visualizer = Visualizer(self.logger)
    
    def run_exploratory_analysis(self) -> pd.DataFrame:
        """
        Run exploratory data analysis: load data, prepare target and sequence features,
        then create population features and generate EDA plots.
        """
        df = self.data_loader.load_data()
        df = self.preprocessor.prepare_target(df)
        df = self.preprocessor.extract_sequence_features(df)
        df_pop = self.preprocessor.create_population_features(df.copy())
        self.visualizer.plot_eda(df_pop)
        return df_pop
    
    

# ------------------------------------------------------------------------
    def run_cross_validation(self, df):
        """Run detailed cross-validation with multiple models and scalers,
        displaying a live progress bar."""
        # ---------- resume from checkpoint ----------------------------------
        cv_cp = self.load_checkpoint("cross_validation")
        if cv_cp is not None:
            self.logger.info("Resuming cross-validation from checkpoint")
            return cv_cp

        # ---------- data ----------------------------------------------------
        X = df.drop(
            ['clinical_significance', 'mapped_target', 'target_encoded',
            ],
            axis=1, errors='ignore'
        )
        y = df['target_encoded']

        # ---------- bookkeeping for tqdm ------------------------------------
        scaler_names = list(self.preprocessor.scalers)           # e.g. ['none','standard',...]
        model_names  = ['RandomForest', 'XGBoost',
                        'LogisticRegression', 'FairnessAware']
        p_total      = len(scaler_names) * len(model_names)      # combos to run

        results = []
        with tqdm(total=p_total, desc="Cross-val combos", unit="combo") as pbar:
            for scaler_name in scaler_names:
                num_cols, cat_cols, _ = self.preprocessor.prepare_column_lists(
                    df, population_aware=True
                )
                # drop the columns that will be handled elsewhere
                num_cols = [c for c in num_cols if c not in ('allele', 'allele_string')]
                cat_cols = [c for c in cat_cols if c not in
                            ('allele', 'allele_string', 'gene', 'consequence_type')]

                pre = self.preprocessor.build_preprocessor(
                    numeric_features=num_cols,
                    categorical_features=cat_cols,
                    sequence_features=['allele', 'allele_string'],          # seq features off for CV speed
                    numeric_scaler=scaler_name
                )

                group_kfold = GroupKFold(n_splits=5)
                groups = df['population_name'] if 'population_name' in df.columns else None
                cv_scheme = group_kfold if groups is not None else 5

                for model_name in model_names:
                    pipe = Pipeline([
                        ('pathway', PathwayFeatureExtractor()),
                        ('pre',     pre),
                        ('clf',     self.model_manager.get_model_instance(model_name))
                    ])

                    cv_res = cross_validate(
                        pipe, X, y,
                        cv=cv_scheme, groups=groups,
                        scoring=['accuracy', 'precision_weighted',
                                'recall_weighted', 'f1_weighted'],
                        n_jobs=-1, error_score='raise'
                    )

                    results.append({
                        'scaler': scaler_name,
                        'model':  model_name,
                        'acc':    cv_res['test_accuracy'].mean(),
                        'prec':   cv_res['test_precision_weighted'].mean(),
                        'rec':    cv_res['test_recall_weighted'].mean(),
                        'f1':     cv_res['test_f1_weighted'].mean()
                    })

                    pbar.update(1)            # ➋ tick progress bar

        cv_df = pd.DataFrame(results)
        cv_df.to_csv('full_cv_metrics.csv', index=False)
        self.logger.info("Saved full CV metrics → full_cv_metrics.csv")

        self.save_checkpoint("cross_validation", cv_df)
        self.visualizer.plot_cv_results(cv_df)
        return cv_df


    
    def run_model_comparison(self, df,
                         model_names=('RandomForest', 'XGBoost', 'LogisticRegression'),
                         random_state: int = 42):
        """
        Compare population-aware vs. non-population-aware pipelines.
        Saves/loads checkpoints for every heavy step.
        """
        if isinstance(model_names, str):
            model_names = [model_names]

        self.logger.info(f"\n{'='*80}\nStarting analysis for models: "
                        f"{', '.join(model_names)}\n{'='*80}")

        # ------------------------------------------------------------------
        # 0. Common data preparation
        df_seq   = self.preprocessor.extract_sequence_features(df)
        df_pop   = self.preprocessor.create_population_features(df_seq.copy())
        df_npop  = df_seq.copy()

        pop_num,  pop_cat,  pop_seq  = self.preprocessor.prepare_column_lists(df_pop,  population_aware=True)
        npo_num,  npo_cat,  npo_seq  = self.preprocessor.prepare_column_lists(df_npop, population_aware=False)

        # ------------------------------------------------------------------
        # 1. Train-test split (or load)
        split_cp = self.load_checkpoint("model_comparison_split")
        if split_cp:
            self.logger.info("Resuming from data-split checkpoint")
            X_tr, X_te = split_cp['X_train'], split_cp['X_test']
            y_tr, y_te = split_cp['y_train'], split_cp['y_test']
            pop_tr, pop_te = split_cp['pop_train'], split_cp['pop_test']
        else:
            X_tr, X_te, y_tr, y_te, pop_tr, pop_te = (
                self.preprocessor.population_stratified_split(
                    df_pop, test_size=0.2, random_state=random_state)
            )
            self.save_checkpoint("model_comparison_split", {
                'X_train': X_tr, 'X_test': X_te,
                'y_train': y_tr, 'y_test': y_te,
                'pop_train': pop_tr, 'pop_test': pop_te
            })

        # separate population cols for the non-pop arm
        pop_cols  = [c for c in X_tr.columns if c == 'population_name' or c.startswith('pop_')]
        X_tr_pop, X_te_pop = X_tr.copy(), X_te.copy()
        X_tr_npo = X_tr.drop(pop_cols, axis=1, errors='ignore')
        X_te_npo = X_te.drop(pop_cols, axis=1, errors='ignore')

        pre_pop = self.preprocessor.build_preprocessor(pop_num, pop_cat, pop_seq)
        pre_npo = self.preprocessor.build_preprocessor(npo_num, npo_cat, npo_seq)

        # ------------------------------------------------------------------
        all_results = {}

        for model_name in model_names:
            tag = model_name.lower()
            self.logger.info(f"\n***** MODEL = {model_name} *****")

            # --------------- Population-aware arm -----------------
            cp_pop = self.load_checkpoint(f"model_comparison_pop_{tag}")
            if cp_pop:
                self.logger.info("Resuming pop-aware checkpoint")
                pop_results = cp_pop['pop_results']
                # feature names come from stored ColumnTransformer
                feat_pop = pop_results['model'].__dict__.get('_feature_names', None)
                if feat_pop is None and 'pre' in cp_pop:         # backward-compat
                    feat_pop = self.preprocessor.get_feature_names(cp_pop['pre'])
            else:
                pipe_pop = Pipeline([
                    ('pathway', PathwayFeatureExtractor()),
                    ('pre',     pre_pop),
                    ('clf',     self.model_manager.get_model_instance(model_name, random_state))
                ])
                pipe_pop.fit(X_tr_pop, y_tr)

                feat_pop = self.preprocessor.get_feature_names(pipe_pop.named_steps['pre'])
                pop_results = self.model_manager.train_evaluate_model(
                    pipe_pop.named_steps['pre'].transform(X_tr_pop), y_tr,
                    pipe_pop.named_steps['pre'].transform(X_te_pop), y_te,
                    pop_test=pop_te,
                    estimator=pipe_pop.named_steps['clf'],
                    random_state=random_state
                )
                # store feature names for later plotting
                pop_results['model']._feature_names = feat_pop
                self.save_checkpoint(f"model_comparison_pop_{tag}", {'pop_results': pop_results})

            # --------------- Non-population-aware arm ------------
            cp_npo = self.load_checkpoint(f"model_comparison_nonpop_{tag}")
            if cp_npo:
                self.logger.info("Resuming non-pop checkpoint")
                npo_results = cp_npo['nonpop_results']
                feat_npo = npo_results['model'].__dict__.get('_feature_names', None)
                if feat_npo is None and 'pre' in cp_npo:
                    feat_npo = self.preprocessor.get_feature_names(cp_npo['pre'])
            else:
                pipe_npo = Pipeline([
                    ('pathway', PathwayFeatureExtractor()),
                    ('pre',     pre_npo),
                    ('clf',     self.model_manager.get_model_instance(model_name, random_state))
                ])
                pipe_npo.fit(X_tr_npo, y_tr)

                feat_npo = self.preprocessor.get_feature_names(pipe_npo.named_steps['pre'])
                npo_results = self.model_manager.train_evaluate_model(
                    pipe_npo.named_steps['pre'].transform(X_tr_npo), y_tr,
                    pipe_npo.named_steps['pre'].transform(X_te_npo), y_te,
                    pop_test=pop_te,
                    estimator=pipe_npo.named_steps['clf'],
                    random_state=random_state
                )
                npo_results['model']._feature_names = feat_npo
                self.save_checkpoint(f"model_comparison_nonpop_{tag}", {'nonpop_results': npo_results})

            # --------------- Visualisation & summary --------------
            self.visualizer.visualize_results(pop_results, npo_results)

            if feat_pop and hasattr(pop_results['model'], 'feature_importances_'):
                self.visualizer.feature_importance_plot(
                    pop_results['model'], f"{model_name} (Pop-Aware)", feat_pop)

            if feat_npo and hasattr(npo_results['model'], 'feature_importances_'):
                self.visualizer.feature_importance_plot(
                    npo_results['model'], f"{model_name} (Non-Pop)", feat_npo)

            all_results[model_name] = {'pop': pop_results, 'nonpop': npo_results}

        self.save_checkpoint("model_comparison_final", all_results)
        self.logger.info(f"\n{'='*80}\nAnalysis Complete\n{'='*80}")
        return all_results



    
    def run_hyperparameter_tuning(self, df, model_name='XGBoost', model_names=None):
        """Run hyper-parameter tuning for selected model(s)"""
        self.logger.info(f"\n=== Hyperparameter Tuning ===")
        # Checkpoint resume
        # If multiple models provided, only loop through them
        models_to_tune = [model_name]
        if model_names is not None:
            models_to_tune = list(model_names) if isinstance(model_names, (list, tuple)) else [model_names]
        best_results = {}
        for m_name in models_to_tune:
            cp = self.load_checkpoint(f"hyperparam_tuning_{m_name}")
            if cp:
                self.logger.info(f"Resuming hyperparameter tuning for {m_name} from checkpoint")
                best_estimator, best_params, best_score = cp['best_estimator'], cp['best_params'], cp['best_score']
            else:
                X = df.drop(['clinical_significance', 'mapped_target', 'target_encoded'], axis=1, errors='ignore')
                y = df['target_encoded']
                num_cols, cat_cols, seq_cols = self.preprocessor.prepare_column_lists(df, population_aware=True)
                pre = self.preprocessor.build_preprocessor(num_cols, cat_cols, seq_cols, numeric_scaler='standard')
                
                if m_name == 'XGBoost':
                    base_model = self.model_manager.get_model_instance('XGBoost')
                    pipe = Pipeline([('pre', pre), ('clf', base_model)])
                    param_grid = {
                        'clf__n_estimators': [100, 200],
                        'clf__max_depth': [3, 5, 7],
                        'clf__learning_rate': [0.01, 0.1],
                        'clf__reg_alpha': [0, 1],
                        'clf__reg_lambda': [1, 10]
                    }
                elif m_name == 'RandomForest':
                    base_model = self.model_manager.get_model_instance('RandomForest')
                    pipe = Pipeline([('pre', pre), ('clf', base_model)])
                    param_grid = {
                        'clf__n_estimators': [50, 100, 200],
                        'clf__max_depth': [5, 10, 15],
                        'clf__min_samples_split': [2, 5, 10],
                        'clf__min_samples_leaf': [1, 2, 4]
                    }
                elif m_name == 'LogisticRegression':
                    base_model = self.model_manager.get_model_instance('LogisticRegression')
                    pipe = Pipeline([('pre', pre), ('clf', base_model)])
                    param_grid = {
                        'clf__C': [0.1, 1.0, 10.0],
                        'clf__solver': ['liblinear', 'saga'],
                        'clf__penalty': ['l1', 'l2']
                    }
                else:
                    raise ValueError(f"Tuning not configured for model: {m_name}")
                
                # Prepare training split with population stratification
                X_train, X_test, y_train, y_test, pop_train, pop_test = self.preprocessor.population_stratified_split(df, test_size=0.2, random_state=42)
                groups_train = pop_train if pop_train is not None else None
                
                group_kfold = GroupKFold(n_splits=5)
                search = RandomizedSearchCV(
                    pipe,
                    param_distributions=param_grid,
                    n_iter=20,
                    cv=group_kfold if groups_train is not None else 5,
                    scoring='accuracy',
                    n_jobs=-1,
                    verbose=0,
                    random_state=42
                )
                search.fit(X_train, y_train, groups=groups_train)
                self.logger.info(f"Best parameters for {m_name}: {search.best_params_}")
                self.logger.info(f"Best CV accuracy for {m_name}: {search.best_score_:.4f}")
                
                best_estimator, best_params, best_score = search.best_estimator_, search.best_params_, search.best_score_
                # Save checkpoint
                self.save_checkpoint(f"hyperparam_tuning_{m_name}", {
                    'best_estimator': best_estimator,
                    'best_params': best_params,
                    'best_score': best_score
                })
            best_results[m_name] = (best_estimator, best_params, best_score)
        # Return results
        # If single model, unpack tuple
        if len(models_to_tune) == 1:
            return best_estimator, best_params, best_score
        return best_results


    
    def run_holdout_evaluation(self, df, best_models=None):
        """Hold-out eval comparing pop-aware vs non-pop pipelines.

        If y has >2 unique classes the fairness reducer is skipped because
        DemographicParity/EqualizedOdds in `fairlearn` accept only binary labels.
        """
        self.logger.info("\n=== Hold-out Evaluation ===")

        # ------------------------------------------------------------------
        # 0. split chronologically, preserve population stratification
        # ------------------------------------------------------------------
        X_tr, X_te, y_tr, y_te, pop_tr, pop_te = (
            self.preprocessor.population_stratified_split(
                df, test_size=0.20, random_state=42
            )
        )

        pop_cols = [c for c in X_tr.columns
                    if c == "population_name" or c.startswith("pop_")]
        X_tr_no = X_tr.drop(pop_cols, axis=1, errors="ignore")
        X_te_no = X_te.drop(pop_cols, axis=1, errors="ignore")

        if best_models is None:
            best_models = {
                "population_aware":    {"scaler": "standard", "model": "RandomForest"},
                "nonpopulation_aware": {"scaler": "standard", "model": "RandomForest"},
            }

        multiclass = len(np.unique(y_tr)) > 2
        if multiclass:
            self.logger.info(
                "Detected multiclass target → fairness reductions will be skipped."
            )

        results = {}

        # ------------------------------------------------------------------
        # 1. evaluate both arms
        # ------------------------------------------------------------------
        for mode, (Xtr, Xte, popte) in [
            ("population_aware",    (X_tr,    X_te,    pop_te)),
            ("nonpopulation_aware", (X_tr_no, X_te_no, pop_te)),
        ]:
            self.logger.info(f"\n=== Hold-out eval: {mode} ===")

            scaler_name = best_models[mode]["scaler"]
            model_name  = best_models[mode]["model"]

            num, cat, seq = self.preprocessor.prepare_column_lists(
                df, population_aware=(mode == "population_aware")
            )
            pre = self.preprocessor.build_preprocessor(
                num, cat, seq, numeric_scaler=scaler_name
            )

            base_est = self.model_manager.get_model_instance(model_name)

            if not multiclass:
                # ------------- binary task → add fairness constraint ---------
                fair_est = GridSearch(
                    estimator=base_est,
                    constraints=DemographicParity(),
                    grid_size=20,
                )
                final_est = fair_est
                sens_train = pop_tr.values if pop_tr is not None else None
                fit_kwargs = {"clf__sensitive_features": sens_train}
            else:
                # ------------- multiclass → plain estimator ------------------
                final_est = base_est
                fit_kwargs = {}

            pipe = Pipeline([
                ("pathway", PathwayFeatureExtractor()),
                ("pre",     pre),
                ("clf",     final_est),
            ])
            pipe.fit(Xtr, y_tr, **fit_kwargs)

            # predict (pass sensitive_features only if fairness active)
            predict_kwargs = {}
            if not multiclass:
                predict_kwargs["clf__sensitive_features"] = (
                    popte.values if popte is not None else None
                )
            y_pred = pipe.predict(Xte, **predict_kwargs)
            
            # -------------------- reporting -------------------------------
            self.logger.info(f"\n>> {model_name} + {scaler_name}")
            self.logger.info(classification_report(
                y_te, y_pred,
                target_names=["benign", "pathogenic", "uncertain"]
            ))

            if popte is not None:
                acc_by_pop = {
                    pop: accuracy_score(y_te[popte == pop], y_pred[popte == pop])
                    for pop in popte.unique()
                }
                self.logger.info(f"Per-population accuracy: {acc_by_pop}")
            else:
                acc_by_pop = None

            # optional ACMG + ML hybrid score (unchanged)
            acmg_int = ACMGPopulationIntegration()
            if hasattr(pipe, "predict_proba"):
                ml_probs = pipe.predict_proba(Xte, **predict_kwargs)[:, 1]
            else:
                ml_probs = None

            if ml_probs is not None:
                criteria_lists = (
                    df.loc[X_te.index, "acmg_criteria"]
                    if "acmg_criteria" in df.columns else
                    [[] for _ in range(len(y_pred))]
                )
                hybrid = acmg_int.integrate_with_ml_model(
                    ml_probs, criteria_lists,
                    popte if popte is not None else [None]*len(y_pred)
                )
            else:
                hybrid = None

            results[mode] = {
                "pipeline":            pipe,
                "accuracy":            accuracy_score(y_te, y_pred),
                "population_accuracy": acc_by_pop,
                "hybrid_scores":       hybrid,
            }

        # ------------------------------------------------------------------
        # 2. checkpoint & return
        # ------------------------------------------------------------------
        self.save_checkpoint("holdout_evaluation", results)
        return results





    
    def save_checkpoint(self, stage: str, data: dict):
        """
        Save a checkpoint dictionary to a file (timestamped) and update a 'latest' pointer.
        """
        import os
        from datetime import datetime
        
        checkpoint_dir = os.path.join("checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = os.path.join(checkpoint_dir, f"{stage}_{timestamp}.pkl")
        joblib.dump(data, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        latest_path = os.path.join(checkpoint_dir, f"{stage}_latest.txt")
        with open(latest_path, 'w') as f:
            f.write(checkpoint_path)
    
    def load_checkpoint(self, stage: str):
        """
        Load the most recent checkpoint for a given stage (if available).
        """
        import os
        latest_path = os.path.join("checkpoints", f"{stage}_latest.txt")
        if os.path.exists(latest_path):
            with open(latest_path, 'r') as f:
                path = f.read().strip()
            if os.path.exists(path):
                self.logger.info(f"Loading checkpoint from {path}")
                return joblib.load(path)
        self.logger.info(f"No checkpoint found for stage: {stage}")
        return None


def main():
    """Main execution function with error handling and checkpointing"""
    FILE_PATH = "raw_variant_data.csv"
    logger = Logger()

    try:
        analysis = GeneticVariantAnalysis(FILE_PATH, logger)

        # ------------------------------------------------------------------ #
        # 1. Exploratory data analysis
        # ------------------------------------------------------------------ #
        logger.info("\n=== Running Exploratory Data Analysis ===")
        try:
            df_pop = analysis.run_exploratory_analysis()

            # ------------------------------------------------------------------ #
            # 2. Cross-validation
            # ------------------------------------------------------------------ #
            logger.info("\n=== Running Cross-Validation ===")
            try:
                cv_results = analysis.run_cross_validation(df_pop)

                # ------------------------------------------------------------------ #
                # 3. Model-family comparison (pop-aware vs non-pop-aware)
                # ------------------------------------------------------------------ #
                logger.info("\n=== Running Model Comparison ===")
                try:
                    analysis.run_model_comparison(
                        df_pop,
                        model_names=('RandomForest', 'XGBoost', 'LogisticRegression')
                    )

                    # ------------------------------------------------------------------ #
                    # 4. Hyper-parameter tuning (already checkpointed)
                    # ------------------------------------------------------------------ #
                    logger.info("\n=== Running Hyperparameter Tuning ===")
                    try:
                        analysis.run_hyperparameter_tuning(
                            df_pop,
                            model_names=('RandomForest', 'XGBoost', 'LogisticRegression')
                        )

                        # ------------------------------------------------------------------ #
                        # 5. Hold-out evaluation ─ run once **per model family**
                        # ------------------------------------------------------------------ #
                        logger.info("\n=== Running Hold-out Evaluation ===")
                        try:
                            for winner in ("RandomForest",
                                           "XGBoost",
                                           "LogisticRegression"):

                                best_models = {
                                    "population_aware": {
                                        "scaler": "standard",
                                        "model":  winner
                                    },
                                    "nonpopulation_aware": {
                                        "scaler": "standard",
                                        "model":  winner
                                    },
                                }

                                logger.info(
                                    f"\n*** Hold-out results for {winner} ***"
                                )
                                analysis.run_holdout_evaluation(
                                    df_pop,
                                    best_models=best_models,
                                )

                        except Exception as e:
                            logger.error(f"Error in hold-out evaluation: {e}")
                            import traceback; logger.error(traceback.format_exc())

                    except Exception as e:
                        logger.error(f"Error in hyper-parameter tuning: {e}")
                        import traceback; logger.error(traceback.format_exc())
                except Exception as e:
                    logger.error(f"Error in model comparison: {e}")
                    import traceback; logger.error(traceback.format_exc())
            except Exception as e:
                logger.error(f"Error in cross-validation: {e}")
                import traceback; logger.error(traceback.format_exc())
        except Exception as e:
            logger.error(f"Error in exploratory analysis: {e}")
            import traceback; logger.error(traceback.format_exc())
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback; logger.error(traceback.format_exc())

    logger.info("\n=== Analysis Complete ===")


if __name__ == "__main__":
    main()