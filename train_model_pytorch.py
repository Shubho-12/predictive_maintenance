# src/train_model_pytorch.py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from src.utils import set_seed, ensure_dirs, get_logger

set_seed(42)
logger = get_logger(log_file="logs/train_pytorch.log")

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "data" / "raw" / "simulated_sensor_data.csv"
MODELS_DIR = ROOT / "models"
ensure_dirs(MODELS_DIR, ROOT / "logs")

# Parameters
WINDOW = 50        # sequence length (timesteps)
HORIZON = 10       # label horizon (failure within next HORIZON)
BATCH_SIZE = 64
EPOCHS = 6
LR = 1e-3
N_WORKERS = 0

SENSOR_COLS = ['temperature', 'vibration', 'pressure', 'rpm']

class SequenceDataset(Dataset):
    def __init__(self, df, window=WINDOW, horizon=HORIZON):
        self.samples = []
        for mid, g in df.groupby('machine_id'):
            g = g.sort_values('timestamp').reset_index(drop=True)
            N = len(g)
            for end in range(window, N - horizon):
                seq = g.iloc[end - window:end][SENSOR_COLS].values.astype(np.float32)
                future = g.iloc[end:end + horizon]
                label = 1 if future['failure'].sum() > 0 else 0
                self.samples.append((seq, label))
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        seq, label = self.samples[idx]
        return torch.tensor(seq), torch.tensor(label, dtype=torch.long)

class LSTMModel(nn.Module):
    def __init__(self, n_features=4, hidden=64, n_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_features, hidden_size=hidden, num_layers=n_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return torch.sigmoid(out).squeeze(-1)

def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch])
    ys = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return xs, ys

def train():
    if not RAW_CSV.exists():
        logger.error(f"Raw CSV missing at {RAW_CSV}. Run data generator first.")
        return

    df = pd.read_csv(RAW_CSV)
    # Normalize per sensor column (simple global scaler)
    for c in SENSOR_COLS:
        mean, std = df[c].mean(), df[c].std()
        df[c] = (df[c] - mean) / (std + 1e-9)

    # Split machines by id for train/test machine-level split
    machine_ids = sorted(df['machine_id'].unique())
    split = int(0.8 * len(machine_ids))
    train_ids = set(machine_ids[:split])
    train_df = df[df['machine_id'].isin(train_ids)].reset_index(drop=True)
    test_df = df[~df['machine_id'].isin(train_ids)].reset_index(drop=True)

    train_ds = SequenceDataset(train_df)
    test_ds = SequenceDataset(test_df)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=N_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=N_WORKERS)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMModel(n_features=len(SENSOR_COLS)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCELoss()

    logger.info(f"Training on device: {device}; train samples: {len(train_ds)}; test samples: {len(test_ds)}")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = loss_fn(preds, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / max(len(train_ds), 1)
        logger.info(f"Epoch {epoch}/{EPOCHS} - Train loss: {avg_loss:.4f}")

        # Simple eval
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                preds_label = (preds > 0.5).float()
                correct += (preds_label == yb).sum().item()
                total += yb.size(0)
        acc = correct / total if total > 0 else 0.0
        logger.info(f"Epoch {epoch} - Test acc: {acc:.4f}")

    # Save model
    torch.save(model.state_dict(), MODELS_DIR / "lstm_model.pt")
    logger.info("Saved LSTM model to models/lstm_model.pt")

if __name__ == "__main__":
    train()
