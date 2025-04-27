# candle_predictor.py
"""
Candle Direction Predictor
==========================

Workflow implemented:
1. EDA / feature preparation (minimal in this first version).
2. Chronological train / validation / test split.
3. Model exploration with three families (RandomForest, XGBoost, LSTM).
4. Hyper‑parameter tuning (RandomisedSearchCV for tree models, Optuna for LSTM).
5. Final training on full train+val, evaluation on hold‑out test.
6. Per‑candle maximum adverse excursion (MAE) calculated for trade‑entry sizing.

Supported instruments & default intervals
----------------------------------------
* GBPUSD  – one‑hour candles  (ticker "GBPUSD=X")
* XAUUSD  – 15‑minute candles (ticker "XAUUSD=X")
* ^DJI    – 15‑minute candles (ticker "^DJI")

If Yahoo Finance volume column is missing (all zeros/NaNs) the volume feature is dropped.
If you already have richer CSVs (with real tick volume) place them under `data/` and
name them `TICKER.csv` – the loader will use those first.

Requirements
------------
```
pip install pandas numpy yfinance scikit-learn xgboost optuna tqdm tensorflow
```
(The LSTM branch requires TensorFlow ≥2.10; comment it out otherwise.)
"""

from __future__ import annotations
import os
from datetime import datetime, timedelta
import warnings
from typing import List, Dict, Tuple, Optional

import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import yfinance as yf

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from pathlib import Path
import pandas as pd
try:
    import optuna
    from tensorflow.keras import layers, models, callbacks
except ImportError:
    optuna = None  # LSTM branch will be skipped

# ---------------------------------------------------------------------------
# Global instrument map & pip/point sizing
# ---------------------------------------------------------------------------
INSTRUMENTS = {
    "GBPUSD=X": {"name": "GBPUSD", "interval": "60m", "pip": 0.0001},
    "XAUUSD=X": {"name": "XAUUSD", "interval": "15m", "pip": 0.1},
    "^DJI":      {"name": "DOW30", "interval": "15m", "pip": 1.0},
}

ONE_YEAR_AGO = (datetime.utcnow() - timedelta(days=365)).strftime("%Y-%m-%d")

# ---------------------------------------------------------------------------
# 1. Data loader
# ---------------------------------------------------------------------------


class MarketDataLoader:
    """
    Always reads OHLC[V] data from a local CSV.

    Expected filename pattern
    -------------------------
        <TICKER>_<INTERVAL>.csv
          e.g.  GBPUSD=X_1h.csv
                XAUUSD=X_15m.csv
                ^DJI_15m.csv

    The file can cover ANY date range (2020-today, etc.).
    Must have columns:
        Datetime, Open, High, Low, Close [, Volume]
    Volume is optional; if missing we fill it with zeros.
    """

    def __init__(self, csv_dir: str = "data"):
        self.csv_dir = Path(csv_dir)

    def load(self, ticker: str, interval: str) -> pd.DataFrame:
        path = self.csv_dir / f"{ticker}_{interval}.csv"
        if not path.exists():
            raise FileNotFoundError(f"CSV not found at {path}")

        df = pd.read_csv(path)
        df.rename(columns=lambda c: c.strip(), inplace=True)      # trim spaces
        if "Datetime" not in df.columns:
            df.rename(columns={df.columns[0]: "Datetime"}, inplace=True)
        if {"Date", "Time"}.issubset(df.columns):
            df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
            df.drop(columns=["Date", "Time"], inplace=True)
        else:
            # first column was already parsed as Datetime via parse_dates=[0]
            df["Datetime"] = pd.to_datetime(df.iloc[:,0])

        df.set_index("Datetime", inplace=True)

        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                raise ValueError(f"{col} column missing in {path}")

        if "Volume" not in df.columns:
            df["Volume"] = 0

        return df

# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------
class FeatureEngineer:
    BASIC_FEATURES = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(self, drop_volume_if_missing: bool = True):
        self.drop_volume_if_missing = drop_volume_if_missing

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        # Replace missing volume with zeros and optionally drop
        if self.drop_volume_if_missing and (df["Volume"].sum() == 0 or df["Volume"].isna().all()):
            df.drop(columns=["Volume"], inplace=True)
        # Derived candle geometry
        df["Body"] = df["Close"] - df["Open"]
        df["UpperWick"] = df["High"] - df[["Close", "Open"]].max(axis=1)
        df["LowerWick"] = df[["Close", "Open"]].min(axis=1) - df["Low"]
        return df

    @staticmethod
    def make_labels(df: pd.DataFrame) -> pd.Series:
        diff = df["Close"].diff().shift(-1)  # next candle movement
        labels = diff.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        labels = labels.iloc[:-1]  # last candle has no future label
        return labels

# ---------------------------------------------------------------------------
# 3. Model trainer / tuner
# ---------------------------------------------------------------------------
class ModelTrainer:
    def __init__(self):
        self.models_: Dict[str, object] = {}

    # ---------------------- tree models ------------------------
    def tune_tree(self, X_train, y_train, base: str) -> object:
        if base == "RandomForest":
            model = RandomForestClassifier(random_state=42)
            param_dist = {
                "n_estimators": [100, 300, 500],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 2, 4],
            }
        elif base == "XGBoost":
            model = xgb.XGBClassifier(
                objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", random_state=42,
            )
            param_dist = {
                "n_estimators": [200, 400, 600],
                "max_depth": [3, 5, 7],
                "learning_rate": [0.01, 0.05, 0.1],
                "subsample": [0.7, 1.0],
            }
        else:
            raise ValueError(base)

        search = RandomizedSearchCV(
            model, param_distributions=param_dist, n_iter=15,
            cv=3, scoring="accuracy", n_jobs=-1, verbose=0, random_state=42,
        )
        search.fit(X_train, y_train)
        return search.best_estimator_

    # ---------------------- LSTM -------------------------------
    def tune_lstm(self, X_train: np.ndarray, y_train: np.ndarray) -> Optional[models.Model]:
        if optuna is None:
            warnings.warn("Optuna or TensorFlow not installed; skipping LSTM")
            return None

        n_features = X_train.shape[-1]

        def build_model(trial):
            units = trial.suggest_categorical("units", [32, 64, 128])
            drop  = trial.suggest_float("drop", 0.0, 0.5)
            model = models.Sequential([
                layers.Input(shape=X_train.shape[1:]),
                layers.LSTM(units, return_sequences=False),
                layers.Dropout(drop),
                layers.Dense(3, activation="softmax"),
            ])
            model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
            return model

        def objective(trial):
            model = build_model(trial)
            es = callbacks.EarlyStopping(patience=3, restore_best_weights=True)
            history = model.fit(
                X_train, y_train, validation_split=0.2, epochs=20, batch_size=256,
                callbacks=[es], verbose=0
            )
            return max(history.history["val_accuracy"])

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=15, timeout=1800)
        best_model = build_model(study.best_trial)
        best_model.fit(X_train, y_train, epochs=20, batch_size=256, verbose=0)
        return best_model

# ---------------------------------------------------------------------------
# 4. Evaluation helpers
# ---------------------------------------------------------------------------
class Evaluator:
    def __init__(self, pip_size: float):
        self.pip = pip_size

    def metrics(self, y_true, y_pred) -> Dict[str, float]:
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
        return dict(accuracy=acc, precision=prec, recall=rec, f1=f1)

    def mae(self, df: pd.DataFrame, preds: pd.Series) -> float:
        maes = []
        for idx, pred in preds.items():
            open_p = df.loc[idx, "Open"]
            if pred == 1:  # bullish
                adverse = open_p - df.loc[idx, "Low"]
            elif pred == -1:  # bearish
                adverse = df.loc[idx, "High"] - open_p
            else:
                adverse = 0.0
            maes.append(abs(adverse) / self.pip)
        return float(np.mean(maes))

# ---------------------------------------------------------------------------
# 5. Orchestrator
# ---------------------------------------------------------------------------
class CandlePredictor:
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.info = INSTRUMENTS[ticker]
        self.loader = MarketDataLoader()
        self.fe = FeatureEngineer()
        self.trainer = ModelTrainer()
        self.eval_helper = Evaluator(self.info["pip"])

    def run(self):
        # 1. load & engineer
        # in CandlePredictor.run()
        raw = self.loader.load(self.ticker, self.info["interval"])

        feat = self.fe.transform(raw)
        labels = self.fe.make_labels(feat)
        feat = feat.iloc[:-1]  # align with labels

        # 2. split chrono
        X_temp, X_test, y_temp, y_test = train_test_split(
            feat, labels, test_size=0.15, shuffle=False
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.1765, shuffle=False  # 0.1765*0.85 ≈ 0.15
        )

        # 3. scale numeric cols
        scaler = StandardScaler()
        num_cols = [c for c in X_train.columns if X_train[c].dtype != "object"]
        scaler.fit(X_train[num_cols])
        X_train[num_cols] = scaler.transform(X_train[num_cols])
        X_val[num_cols] = scaler.transform(X_val[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])

        # 4. tune models
        print(f"Tuning tree models for {self.info['name']} …")
        rf = self.trainer.tune_tree(pd.get_dummies(X_train), y_train, base="RandomForest")
        xgbc = self.trainer.tune_tree(pd.get_dummies(X_train), y_train, base="XGBoost")
        lstm = None
        if optuna is not None:
            # reshape for LSTM: (samples, timesteps=1, features)
            lstm = self.trainer.tune_lstm(X_train[num_cols].values[:, None, :], y_train.values)

        # 5. evaluate on test
        results = {}
        for name, model, X_feat in [
            ("RandomForest", rf, pd.get_dummies(X_test).reindex(columns=pd.get_dummies(X_train).columns, fill_value=0)),
            ("XGBoost", xgbc, pd.get_dummies(X_test).reindex(columns=pd.get_dummies(X_train).columns, fill_value=0)),
        ]:
            y_pred = model.predict(X_feat)
            results[name] = self.eval_helper.metrics(y_test, y_pred)
            results[name]["mae_pips"] = self.eval_helper.mae(raw.loc[X_test.index], pd.Series(y_pred, index=X_test.index))
            print(f"\n{name} classification report:\n" + classification_report(y_test, y_pred))

        if lstm is not None:
            y_pred = lstm.predict(X_test[num_cols].values[:, None, :]).argmax(axis=1)
            results["LSTM"] = self.eval_helper.metrics(y_test, y_pred)
            results["LSTM"]["mae_pips"] = self.eval_helper.mae(raw.loc[X_test.index], pd.Series(y_pred, index=X_test.index))
            print("\nLSTM classification report:\n" + classification_report(y_test, y_pred))

        # 6. summary
        print("\n=== Summary ===")
        for m, d in results.items():
            print(f"{m}: Acc={d['accuracy']:.3f}  F1={d['f1']:.3f}  MAE={d['mae_pips']:.2f} pips")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    for t in INSTRUMENTS.keys():
        print(f"\n{'='*80}\nRunning pipeline for {INSTRUMENTS[t]['name']} ({t})")
        CandlePredictor(t).run()
