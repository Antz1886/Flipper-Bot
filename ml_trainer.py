import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ml_trainer")

DATA_FILE = "ml_data/historical_setups.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "rf_setup_scorer.pkl")

def train_model():
    if not os.path.exists(DATA_FILE):
        log.error(f"Data file not found: {DATA_FILE}")
        return

    log.info("Loading dataset...")
    df = pd.read_csv(DATA_FILE)
    
    if len(df) < 50:
        log.error("Not enough data to train a model.")
        return
        
    # Preprocessing
    df = df.dropna()
    
    # Encode direction (BUY=1, SELL=0)
    df["direction_encoded"] = df["direction"].apply(lambda x: 1 if x == "BUY" else 0)
    
    # Select features
    features = ["direction_encoded", "ob_size", "atr", "rsi", "price_vs_ema"]
    X = df[features]
    y = df["label"]
    
    log.info(f"Dataset shape: {X.shape}. Class distribution:\n{y.value_counts(normalize=True)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    log.info("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    log.info(f"Validation Accuracy: {acc:.2%}")
    log.info(f"\n{classification_report(y_test, y_pred)}")
    
    # Feature Importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values("Importance", ascending=False)
    log.info(f"\nFeature Importances:\n{feat_imp}")
    
    # Save
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE)
    log.info(f"Model saved to {MODEL_FILE}")

if __name__ == "__main__":
    train_model()
