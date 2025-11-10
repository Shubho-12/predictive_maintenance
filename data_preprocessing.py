# src/data_preprocessing.py
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import build_and_save_features

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "simulated_sensor_data.csv"
PROCESSED = ROOT / "data" / "processed" / "processed_features.csv"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def preprocess_and_save(window=50, horizon=10, test_size=0.2, random_state=42):
    feats = build_and_save_features(str(RAW), str(PROCESSED), window=window, horizon=horizon)
    # drop columns we won't train on
    X = feats.drop(columns=['machine_id', 'end_index', 'end_time', 'label'])
    y = feats['label'].astype(int)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    print("✅ Scaler saved:", MODELS_DIR / "scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y if y.nunique()>1 else None
    )

    # Save processed arrays for quick load
    joblib.dump((X_train, X_test, y_train, y_test), MODELS_DIR / "processed_data.joblib")
    print(f"✅ Preprocessing complete. Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    preprocess_and_save()
