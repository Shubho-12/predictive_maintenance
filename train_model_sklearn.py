# src/train_model_sklearn.py
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

def load_processed():
    X_train, X_test, y_train, y_test = joblib.load(MODELS_DIR / "processed_data.joblib")
    return X_train, X_test, y_train, y_test

def train_and_save():
    X_train, X_test, y_train, y_test = load_processed()
    clf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    print("✅ RandomForest trained.")
    print(classification_report(y_test, preds))
    if probs is not None and len(set(y_test))>1:
        print("ROC-AUC:", roc_auc_score(y_test, probs))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, MODELS_DIR / "rf_model.pkl")
    print("✅ Model saved:", MODELS_DIR / "rf_model.pkl")
    return clf

if __name__ == "__main__":
    train_and_save()
