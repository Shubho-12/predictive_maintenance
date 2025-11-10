# src/feature_engineering.py
import pandas as pd
import numpy as np
from pathlib import Path

def create_rolling_features(df, window=50, horizon=10):
    """
    df: dataframe with columns ['timestamp','machine_id','temperature','vibration','pressure','rpm','failure']
    window: number of past rows to compute stats on
    horizon: number of future rows; label=1 if any failure occurs within horizon after window end
    """
    out_rows = []
    for mid, g in df.groupby('machine_id'):
        g = g.sort_values('timestamp').reset_index(drop=True)
        N = len(g)
        for end in range(window, N - horizon):
            win = g.iloc[end - window:end]
            future = g.iloc[end:end + horizon]
            feats = {'machine_id': mid, 'end_index': end, 'end_time': g.loc[end, 'timestamp']}
            for col in ['temperature', 'vibration', 'pressure', 'rpm']:
                arr = win[col].values.astype(float)
                feats[f'{col}_mean'] = arr.mean()
                feats[f'{col}_std'] = arr.std()
                feats[f'{col}_min'] = arr.min()
                feats[f'{col}_max'] = arr.max()
                feats[f'{col}_trend'] = np.polyfit(np.arange(len(arr)), arr, 1)[0]  # slope
            # label: failure occurs in horizon window
            feats['label'] = 1 if future['failure'].sum() > 0 else 0
            out_rows.append(feats)
    feat_df = pd.DataFrame(out_rows)
    return feat_df

def build_and_save_features(raw_csv_path, out_path, window=50, horizon=10):
    raw = pd.read_csv(raw_csv_path)
    feats = create_rolling_features(raw, window=window, horizon=horizon)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_path, index=False)
    print("✅ Features saved to:", out_path)
    return feats
