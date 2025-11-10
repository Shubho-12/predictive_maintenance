# app/alert_utils.py
from pathlib import Path
import pandas as pd
from datetime import datetime
from src.utils import ensure_dirs, get_logger

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "logs"
ensure_dirs(LOGS)
logger = get_logger(log_file=str(LOGS / "alerts.log"))

ALERT_CSV = LOGS / "alerts.csv"
if not ALERT_CSV.exists():
    pd.DataFrame(columns=["timestamp","machine_id","probability","message"]).to_csv(ALERT_CSV, index=False)

def send_alert(machine_id, probability, message="Failure probability exceeded threshold"):
    row = {"timestamp": datetime.now().isoformat(), "machine_id": machine_id, "probability": probability, "message": message}
    df = pd.read_csv(ALERT_CSV)
    df = df.append(row, ignore_index=True)
    df.to_csv(ALERT_CSV, index=False)
    logger.warning(f"ALERT: machine {machine_id} prob={probability:.3f} - {message}")
    # simulate email / sms by printing
    print(f"[ALERT SENT] machine_id={machine_id} prob={probability:.3f} msg={message}")
