# src/evaluate_model.py
import joblib
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from src.utils import get_logger, ensure_dirs

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
ensure_dirs(REPORTS_DIR, MODELS_DIR, ROOT / "logs")
logger = get_logger(log_file="logs/evaluate.log")

def evaluate():
    processed_path = MODELS_DIR / "processed_data.joblib"
    model_path = MODELS_DIR / "rf_model.pkl"
    if not processed_path.exists() or not model_path.exists():
        logger.error("Missing processed_data.joblib or rf_model.pkl. Run preprocessing & train first.")
        return

    X_train, X_test, y_train, y_test = joblib.load(processed_path)
    clf = joblib.load(model_path)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(REPORTS_DIR / "classification_report.csv", index=True)
    logger.info("Saved classification_report.csv")

    cm = confusion_matrix(y_test, preds)
    pd.DataFrame(cm, index=["tn","fp","fn","tp"][:cm.shape[0]]).to_csv(REPORTS_DIR / "confusion_matrix.csv", index=True)
    logger.info("Saved confusion_matrix.csv")

    if probs is not None and len(set(y_test))>1:
        auc = roc_auc_score(y_test, probs)
        with open(REPORTS_DIR / "metrics.txt", "w") as f:
            f.write(f"ROC-AUC: {auc:.4f}\n")
        logger.info(f"ROC-AUC: {auc:.4f}")

    print("Evaluation complete. Reports in folder:", REPORTS_DIR)

if __name__ == "__main__":
    evaluate()
