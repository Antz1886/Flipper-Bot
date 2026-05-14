"""
ml_trainer.py — AI Model Training Pipeline
Trains a Random Forest classifier with probability calibration
to score SMC trade setups. Uses stratified cross-validation and
guards against overfitting with depth limits and minimum leaf sizes.

v2.0 — Added CalibratedClassifierCV for realistic probability outputs,
       expanded feature set, better data validation, and cross-val reporting.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, brier_score_loss
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_trainer")

DATA_FILE = "ml_data/historical_setups.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "rf_setup_scorer.pkl")

# Expanded feature columns (must match ml_data_collector.py output)
EXPANDED_FEATURES = [
    "direction_encoded", "ob_size_pct", "atr_pct", "rsi", "price_vs_ema",
    "candle_body_pct", "upper_wick_pct", "lower_wick_pct",
    "distance_to_ob_pct", "hour_of_day"
]

# Legacy features for backward compatibility
LEGACY_FEATURES = [
    "direction_encoded", "ob_size_pct", "atr_pct", "rsi", "price_vs_ema"
]


def train_model():
    if not os.path.exists(DATA_FILE):
        log.error(f"Data file not found: {DATA_FILE}")
        return

    log.info("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    
    if len(df) < 100:
        log.error(f"Not enough data to train a reliable model ({len(df)} samples, need >= 100).")
        return
        
    # Preprocessing
    df = df.dropna(subset=["label"])
    
    # Encode direction (BUY=1, SELL=0)
    df["direction_encoded"] = df["direction"].apply(lambda x: 1 if x == "BUY" else 0)
    
    # Determine available features
    expanded_available = all(f in df.columns for f in EXPANDED_FEATURES)
    if expanded_available:
        features = EXPANDED_FEATURES
        log.info("Using EXPANDED feature set (10 features)")
    else:
        features = LEGACY_FEATURES
        log.info("Using LEGACY feature set (5 features)")
    
    # Fill NaN values in features
    for f in features:
        if f in df.columns:
            df[f] = df[f].fillna(0)
    
    X = df[features]
    y = df["label"]
    
    # ── Data Quality Checks ──────────────────────────────────────────────────
    win_rate = y.mean()
    log.info(f"Dataset shape: {X.shape}")
    log.info(f"Class distribution: {win_rate:.1%} wins, {1-win_rate:.1%} losses")
    
    if win_rate > 0.90 or win_rate < 0.10:
        log.warning(
            f"⚠️  Dataset is EXTREMELY skewed ({win_rate:.1%} wins). "
            "Model predictions will be unreliable. Consider collecting more data."
        )
    elif win_rate > 0.75 or win_rate < 0.25:
        log.warning(
            f"⚠️  Dataset is moderately skewed ({win_rate:.1%} wins). "
            "Using class_weight='balanced' to compensate."
        )
    
    # ── Train/Test Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    log.info(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")
    
    # ── Base Model ───────────────────────────────────────────────────────────
    log.info("Training Random Forest Classifier...")
    base_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,            # Shallow to prevent overfitting
        min_samples_leaf=20,    # Prevent memorizing individual samples
        min_samples_split=40,   # Require substantial evidence before splitting
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    
    # ── Cross-Validation ─────────────────────────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(base_model, X_train, y_train, cv=cv, scoring="accuracy")
    log.info(f"Cross-validation accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
    log.info(f"  Per-fold: {[f'{s:.2%}' for s in cv_scores]}")
    
    # ── Calibrated Model ─────────────────────────────────────────────────────
    log.info("Calibrating probabilities with CalibratedClassifierCV...")
    
    # Fit the base model first
    base_model.fit(X_train, y_train)
    
    # Calibrate using cross-validation on training set
    n_cv = min(5, min(y_train.value_counts()))  # Ensure enough samples per fold
    if n_cv >= 2:
        calibrated_model = CalibratedClassifierCV(
            base_model, cv=n_cv, method="isotonic"
        )
        calibrated_model.fit(X_train, y_train)
        final_model = calibrated_model
        log.info(f"✅ Probability calibration applied ({n_cv}-fold)")
    else:
        log.warning("Not enough samples per class for calibration — using raw model.")
        final_model = base_model
    
    # ── Evaluation ───────────────────────────────────────────────────────────
    y_pred = final_model.predict(X_test)
    y_prob = final_model.predict_proba(X_test)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    
    log.info(f"\n{'='*60}")
    log.info(f"TEST SET RESULTS")
    log.info(f"  Accuracy:    {acc:.2%}")
    log.info(f"  Brier Score: {brier:.4f} (lower = better calibrated, 0 = perfect)")
    log.info(f"\n{classification_report(y_test, y_pred)}")
    
    # ── Probability Distribution Check ───────────────────────────────────────
    log.info("Probability distribution on test set:")
    prob_bins = pd.cut(y_prob, bins=[0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    prob_dist = prob_bins.value_counts().sort_index()
    for bin_label, count in prob_dist.items():
        log.info(f"  {bin_label}: {count} samples ({count/len(y_prob):.1%})")
    
    # Check if model produces varied predictions
    if y_prob.min() > 0.7:
        log.warning(
            f"⚠️  Model minimum prediction is {y_prob.min():.2%} — still too confident! "
            "Consider more training data or deeper trees."
        )
    elif y_prob.max() - y_prob.min() < 0.2:
        log.warning(
            f"⚠️  Model prediction range is only {y_prob.max()-y_prob.min():.2%} — "
            "not discriminating well between good and bad setups."
        )
    else:
        log.info(
            f"✅ Model produces varied predictions: "
            f"range [{y_prob.min():.2%}, {y_prob.max():.2%}]"
        )
    
    # ── Feature Importances (from base model) ────────────────────────────────
    importances = base_model.feature_importances_
    feat_imp = pd.DataFrame(
        {"Feature": features, "Importance": importances}
    ).sort_values("Importance", ascending=False)
    log.info(f"\nFeature Importances:\n{feat_imp.to_string(index=False)}")
    
    # ── Save ─────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_model, MODEL_FILE)
    log.info(f"\n✅ Model saved to {MODEL_FILE}")
    log.info(f"   Features: {features}")
    log.info(f"   Total training samples: {len(X_train)}")
    log.info(f"   Calibration: {'Yes' if n_cv >= 2 else 'No'}")

if __name__ == "__main__":
    train_model()
