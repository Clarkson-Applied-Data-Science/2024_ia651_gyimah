# -*- coding: utf-8 -*-
"""
Script: variant_classifier.py

A class-based framework for genetic variant classification that:
- Loads and preprocesses genetic variant data
- Extracts sequence and population-aware features
- Trains and evaluates both population-aware and non-population-aware models
- Visualizes results and feature importance

Author: Simon Gyimah
Date: 2025-04-23
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from sklearn.model_selection import train_test_split, GroupKFold, cross_validate, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from fairlearn.metrics import MetricFrame
from imblearn.over_sampling import RandomOverSampler

# Configure logging to display time, level, and message
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataLoader:
    """Load genetic variant data from a CSV file."""
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.file_path}")
        # Specify categorical columns to reduce memory usage
        dtype_mapping = {
            'gene': 'category',
            'chromosome': 'category',
            'variant_id': 'category',
            'consequence_type': 'category',
            'clinical_significance': 'category',
            'population_name': 'category',
            'region': 'category',
        }
        df = pd.read_csv(self.file_path, dtype=dtype_mapping, low_memory=False)
        # Convert numeric columns safely
        for col in ['allele_freq', 'allele_count', 'sample_size', 'position']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        logger.info(f"Data shape: {df.shape}")
        return df

class FeatureEngineer:
    """Create and transform features for modeling."""
    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map clinical significance into three classes and encode them."""
        logger.info("Preparing target variable")
        df = df.copy()
        def map_label(x):
            if pd.isna(x):
                return np.nan
            lbl = str(x).lower()
            if 'pathogenic' in lbl:
                return 'pathogenic'
            if 'benign' in lbl and 'pathogenic' not in lbl:
                return 'benign'
            if 'uncertain' in lbl or 'conflicting' in lbl:
                return 'uncertain'
            return np.nan
        df['mapped_target'] = df['clinical_significance'].apply(map_label)
        df = df.dropna(subset=['mapped_target']).reset_index(drop=True)
        # Numeric encoding for the classes
        label_map = {'benign': 0, 'pathogenic': 1, 'uncertain': 2}
        df['target_encoded'] = df['mapped_target'].map(label_map)
        logger.info(f"Class counts:\n{df['mapped_target'].value_counts()}\n")
        return df

    def extract_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DNA sequences into numeric features: length, nucleotide counts, GC content."""
        logger.info("Extracting sequence features")
        df = df.copy()
        seq_col = 'allele' if 'allele' in df.columns else 'allele_string' if 'allele_string' in df.columns else None
        if seq_col:
            df[seq_col] = df[seq_col].fillna('').astype(str)
            df['seq_length'] = df[seq_col].str.len()
            for nuc in ['A', 'C', 'G', 'T']:
                df[f'seq_{nuc}_count'] = df[seq_col].str.count(nuc)
            df['seq_GC_content'] = (
                df['seq_G_count'] + df['seq_C_count']
            ) / df['seq_length'].replace(0, 1)
        else:
            logger.warning("No sequence column found.")
        return df

    def create_population_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add population-specific interaction features and relative allele frequency."""
        logger.info("Creating population-aware features")
        df = df.copy()
        if 'population_name' in df.columns and 'consequence_type' in df.columns:
            df['pop_consequence'] = (
                df['population_name'].astype(str) + '_' + df['consequence_type'].astype(str)
            )
        if 'population_name' in df.columns and 'gene' in df.columns:
            df['pop_gene'] = (
                df['population_name'].astype(str) + '_' + df['gene'].astype(str)
            )
        if 'allele_freq' in df.columns and 'population_name' in df.columns:
            pop_means = df.groupby('population_name', observed=False)['allele_freq'].mean().to_dict()
            df['pop_allele_freq_mean'] = df['population_name'].map(pop_means).astype('float64')
            overall_mean = df['allele_freq'].mean()
            df['pop_allele_freq_mean'] = df['pop_allele_freq_mean'].fillna(overall_mean)
            df['allele_freq_rel'] = (
                df['allele_freq'] / df['pop_allele_freq_mean']
            )
        return df

class PreprocessorBuilder:
    """Build ColumnTransformer pipelines for numeric, categorical, and sequence data."""
    def build(self,
              numeric_features: List[str],
              categorical_features: List[str],
              sequence_feature: Optional[str]
    ) -> ColumnTransformer:
        transformers = []
        # Numeric: fill missing, then standardize
        if numeric_features:
            num_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_pipe, numeric_features))
        # Categorical: fill missing, one-hot encode
        if categorical_features:
            cat_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
            transformers.append(('cat', cat_pipe, categorical_features))
        # Sequence: use k-mer count vectorization
        # --- inside PreprocessorBuilder.build ---------------------------------
        if sequence_feature:
            seq_pipe = Pipeline([
                ('imputer', SimpleImputer(strategy='constant', fill_value='')),
                ('flatten', FunctionTransformer(lambda x: x.ravel(), validate=False)),  # NEW
                ('kmer', CountVectorizer(analyzer='char', ngram_range=(3, 6)))
            ])
            transformers.append(('seq', seq_pipe, [sequence_feature]))
# ----------------------------------------------------------------------

        return ColumnTransformer(transformers=transformers)

    def get_feature_names(self, transformer: ColumnTransformer) -> List[str]:
        """Retrieve feature names post-transform (excluding sequence features)."""
        names: List[str] = []
        for name, trans, cols in transformer.transformers_:
            if name == 'seq':
                continue  # skip sequence feature names
            if hasattr(trans, 'named_steps') and 'onehot' in trans.named_steps:
                fnames = trans.named_steps['onehot'].get_feature_names_out(cols)
                names.extend(fnames.tolist())
            else:
                if isinstance(cols, list):
                    names.extend(cols)
        return names

class ModelTrainer:
    """Train and evaluate classifiers with optional oversampling and population weighting."""
    def __init__(self, model_name: str, random_state: int = 42):
        self.model_name = model_name
        self.random_state = random_state
        self.label_map_inv = {0: 'benign', 1: 'pathogenic', 2: 'uncertain'}

    def get_model(self):
        """Instantiate the chosen model."""
        if self.model_name == 'RandomForest':
            return RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=self.random_state
            )
        if self.model_name == 'XGBoost':
            return XGBClassifier(
                use_label_encoder=False,
                eval_metric='mlogloss',
                random_state=self.random_state
            )
        if self.model_name == 'LogisticRegression':
            return LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            )
        raise ValueError(f"Unknown model: {self.model_name}")

    def train(
        self,
        preprocessor: ColumnTransformer,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        pop_train: Optional[pd.Series] = None
    ) -> Tuple[object, np.ndarray, np.ndarray]:
        """
        Train the model, apply oversampling, and optional population weights.
        Returns the trained model and transformed training data.
        """
        # Transform training data
        X_tr = preprocessor.fit_transform(X_train)
        # Oversample to balance classes
        ros = RandomOverSampler(random_state=self.random_state)
        X_res, y_res = ros.fit_resample(X_tr, y_train)
        # Compute population weights if provided
        sample_weights = None
        if pop_train is not None:
            pop_counts = pop_train.value_counts()
            inv_freq = {pop: len(pop_train)/count for pop, count in pop_counts.items()}
            orig_idx = ros.sample_indices_
            weights = pop_train.map(inv_freq).values
            sample_weights = weights[orig_idx]
        # Fit model
        model = self.get_model()
        if sample_weights is not None:
            model.fit(X_res, y_res, sample_weight=sample_weights)
        else:
            model.fit(X_res, y_res)
        return model, X_res, y_res

    def evaluate(
        self,
        model: object,
        transformer: ColumnTransformer,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        pop_test: Optional[pd.Series] = None
    ) -> Dict:
        """
        Evaluate the trained model and print metrics and fairness.
        """
        X_te = transformer.transform(X_test)
        y_pred = model.predict(X_te)
        # Convert numeric to text labels for reporting
        y_test_text = y_test.map(self.label_map_inv)
        y_pred_text = pd.Series(y_pred).map(self.label_map_inv)
        print(classification_report(
            y_test_text,
            y_pred_text,
            labels=list(self.label_map_inv.values())
        ))
        results = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_recall_fscore_support(
                y_test, y_pred, average='weighted')[0],
            'recall': precision_recall_fscore_support(
                y_test, y_pred, average='weighted')[1],
            'f1': precision_recall_fscore_support(
                y_test, y_pred, average='weighted')[2],
            'model': model
        }
        # Fairness metrics by population
        if pop_test is not None:
            mf = MetricFrame(
                metrics=accuracy_score,
                y_true=y_test_text,
                y_pred=y_pred_text,
                sensitive_features=pop_test
            )
            results['fairness'] = mf.by_group.to_dict()
            results['max_accuracy_diff'] = mf.difference()
            print(f"Population fairness: {results['fairness']}")
            print(f"Max accuracy gap: {results['max_accuracy_diff']:.3f}")
        return results

class VariantClassificationApp:
    """Orchestrate the full workflow using the above classes."""
    def __init__(
        self,
        file_path: str,
        model_name: str = 'RandomForest',
        test_size: float = 0.2,
        random_state: int = 42
    ):
        self.file_path = file_path
        self.model_name = model_name
        self.test_size = test_size
        self.random_state = random_state
        self.loader = DataLoader(file_path)
        self.engineer = FeatureEngineer()
        self.builder = PreprocessorBuilder()
        self.trainer = ModelTrainer(model_name, random_state)

    def _get_column_lists(
        self,
        df: pd.DataFrame,
        population_aware: bool
    ) -> Tuple[List[str], List[str], Optional[str]]:
        """Identify numeric, categorical, and sequence columns."""
        numeric = [
            'allele_freq', 'allele_count', 'sample_size', 'position',
            'seq_length', 'seq_A_count', 'seq_C_count', 'seq_G_count',
            'seq_T_count', 'seq_GC_content'
        ]
        numeric = [c for c in numeric if c in df.columns]
        if population_aware and 'allele_freq_rel' in df.columns:
            numeric.append('allele_freq_rel')
        categorical = [
            'gene', 'chromosome', 'consequence_type', 'region'
        ]
        categorical = [c for c in categorical if c in df.columns]
        if population_aware:
            for c in ['population_name', 'pop_consequence', 'pop_gene']:
                if c in df.columns:
                    categorical.append(c)
        sequence_feature = 'allele' if 'allele' in df.columns else 'allele_string' if 'allele_string' in df.columns else None
        return numeric, categorical, sequence_feature

    def run(self):
        # Load and prepare data
        df = self.loader.load_data()
        df = self.engineer.prepare_target(df)
        df = self.engineer.extract_sequence_features(df)
        # Prepare population-aware and non-aware datasets
        df_pop = self.engineer.create_population_features(df.copy())
        df_nonpop = df.copy()
        # Split data
        X = df_pop.drop(columns=['clinical_significance','mapped_target','target_encoded'], errors='ignore')
        y = df_pop['target_encoded']
        pop = df_pop['population_name'] if 'population_name' in df_pop.columns else None
        X_train, X_test, y_train, y_test, pop_train, pop_test = train_test_split(
            X, y, pop,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        # Build and train models
        results = {}
        for mode, df_source in [('Population-Aware', df_pop), ('Non-Population-Aware', df_nonpop)]:
            pop_flag = (mode == 'Population-Aware')
            # Get columns
            numeric, categorical, sequence_feature = self._get_column_lists(
                df_source, pop_flag
            )
            # Build preprocessor
            pre = self.builder.build(numeric, categorical, sequence_feature)
            # Train model
            model, _, _ = self.trainer.train(
                pre,
                X_train if pop_flag else X_train.drop([c for c in X_train.columns if 'pop_' in c or c=='population_name'], axis=1, errors='ignore'),
                y_train,
                pop_train if pop_flag else None
            )
            # Evaluate model
            results[mode] = self.trainer.evaluate(
                model,
                pre,
                X_test if pop_flag else X_test.drop([c for c in X_test.columns if 'pop_' in c or c=='population_name'], axis=1, errors='ignore'),
                y_test,
                pop_test if pop_flag else None
            )
        # Compare overall performance
        for mode in results:
            print(f"{mode} Accuracy: {results[mode]['accuracy']:.3f}")
        # Visual comparison could be added here

if __name__ == '__main__':
    app = VariantClassificationApp(
        file_path='raw_variant_data.csv',
        model_name='RandomForest',
        test_size=0.2,
        random_state=42
    )
    app.run()
