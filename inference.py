# src/inference.py
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PROCESSED_CSV = ROOT / "data" / "processed" / "processed_features.csv"

# sensors and per-sensor features must match src/feature_engineering.create_rolling_features
SENSOR_COLS = ['temperature', 'vibration', 'pressure', 'rpm']
PER_SENSOR_FEATURES = ['mean', 'std', 'min', 'max', 'trend']

# module-level cache
_MODEL = None
_SCALER = None


def _ordered_feature_names() -> List[str]:
    """Return feature names in the exact order used by preprocessing and scaler."""
    names: List[str] = []
    for s in SENSOR_COLS:
        for feat in PER_SENSOR_FEATURES:
            names.append(f"{s}_{feat}")
    return names


def _load_model_and_scaler() -> Tuple[Any, Any]:
    """Load and cache model and scaler from MODELS_DIR."""
    global _MODEL, _SCALER
    model_path = MODELS_DIR / "rf_model.pkl"
    scaler_path = MODELS_DIR / "scaler.pkl"
    if _MODEL is None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}. Please train and save the model first.")
        _MODEL = joblib.load(model_path)
    if _SCALER is None:
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}. Please run preprocessing to save scaler.pkl.")
        _SCALER = joblib.load(scaler_path)
    return _MODEL, _SCALER


def _compute_trend(arr: np.ndarray) -> float:
    """Simple linear trend (slope) for 1D array. Returns 0 for constant-length < 2."""
    if len(arr) < 2:
        return 0.0
    x = np.arange(len(arr))
    # fit degree-1 polynomial
    try:
        slope = np.polyfit(x, arr, 1)[0]
    except Exception:
        slope = float(arr[-1] - arr[0]) / (len(arr) - 1 + 1e-9)
    return float(slope)


def predict_from_list(values_list: List[float]) -> Tuple[int, Any]:
    """
    Accepts a list/array of feature values in the same order as _ordered_feature_names()
    Returns: (pred_int, prob_float_or_None)
    """
    model, scaler = _load_model_and_scaler()
    X = np.array(values_list).reshape(1, -1)
    Xs = scaler.transform(X)
    pred = int(model.predict(Xs)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(Xs)[0][1])
    return pred, prob


def predict_failure(features_dict: Dict[str, Any]) -> Tuple[int, Any]:
    """
    Predict using a dictionary input.

    Supported input formats:
    1) features_dict contains exact engineered feature names as keys:
       e.g. {'temperature_mean':..., 'temperature_std':..., 'vibration_mean':..., ...}

    2) features_dict contains raw sensor arrays for each sensor:
       e.g. {'temperature': [..values..], 'vibration': [..], 'pressure': [..], 'rpm': [..]}
       The function will compute mean/std/min/max/trend for each sensor in the expected order.

    Returns:
      (prediction_int, probability_float_or_None)
    """
    ordered_names = _ordered_feature_names()

    # Case A: keys are engineered feature names
    if all(k in features_dict for k in ordered_names):
        values = [float(features_dict[k]) for k in ordered_names]
        return predict_from_list(values)

    # Case B: keys are raw sensor arrays
    if all(s in features_dict for s in SENSOR_COLS):
        values = []
        for s in SENSOR_COLS:
            arr = np.array(features_dict[s], dtype=float)
            values.append(float(np.mean(arr)))
            values.append(float(np.std(arr)))
            values.append(float(np.min(arr)))
            values.append(float(np.max(arr)))
            values.append(float(_compute_trend(arr)))
        return predict_from_list(values)

    # If neither format matches, raise a helpful error
    raise ValueError(
        "features_dict must contain either all engineered feature names "
        f"({ordered_names[:4]}... ) OR raw sensor arrays for each sensor {SENSOR_COLS}."
    )


if __name__ == "__main__":
    # Quick local demo: try to read processed CSV and predict on last row
    if PROCESSED_CSV.exists():
        import pandas as pd
        df = pd.read_csv(PROCESSED_CSV)
        names = _ordered_feature_names()
        missing = [n for n in names if n not in df.columns]
        if missing:
            print("Processed CSV is missing these feature columns:", missing)
        else:
            sample_vals = df[names].iloc[-1].values.tolist()
            p, pr = predict_from_list(sample_vals)
            print("Prediction:", p, "Probability:", pr)
    else:
        # If processed CSV missing, try demo using raw data if available
        RAW = ROOT / "data" / "raw" / "simulated_sensor_data.csv"
        if RAW.exists():
            import pandas as pd
            raw = pd.read_csv(RAW)
            # pick last window for first machine
            mid = raw['machine_id'].unique()[0]
            sub = raw[raw['machine_id'] == mid].sort_values('timestamp').tail(50)
            demo_input = {c: sub[c].values.tolist() for c in SENSOR_COLS}
            p, pr = predict_failure(demo_input)
            print(f"Demo prediction for machine {mid}: {p} prob={pr}")
        else:
            print("No processed features or raw data found. Run preprocessing or data generator first.")
