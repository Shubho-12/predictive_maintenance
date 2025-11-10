import pytest
import pandas as pd
from pathlib import Path
from src.inference import predict_from_list, _ordered_feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "processed_features.csv"

def test_processed_data_exists():
    """Check if processed dataset exists."""
    assert DATA_PATH.exists(), "Processed features CSV missing. Run preprocessing first."

def test_prediction_shape():
    """Ensure model returns valid prediction and probability."""
    from joblib import load
    names = _ordered_feature_names()
    df = pd.read_csv(DATA_PATH)
    sample = df[names].iloc[0].values.tolist()
    pred, prob = predict_from_list(sample)
    assert pred in [0, 1], "Prediction must be binary."
    assert 0.0 <= prob <= 1.0, "Probability must be between 0 and 1."

def test_model_files_exist():
    """Ensure trained models are saved."""
    model_dir = PROJECT_ROOT / "models"
    assert (model_dir / "rf_model.pkl").exists(), "RandomForest model missing."
    assert (model_dir / "scaler.pkl").exists(), "Scaler file missing."
