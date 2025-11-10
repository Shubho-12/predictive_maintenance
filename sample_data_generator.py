# data/sample_data_generator.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT.mkdir(parents=True, exist_ok=True)

def generate_sensor_data(num_machines=5, num_records=1500, anomaly_chance=0.005):
    data = []
    start_time = datetime.now()

    for machine_id in range(1, num_machines + 1):
        temperature = random.uniform(50, 70)
        vibration = random.uniform(0.3, 0.6)
        pressure = random.uniform(1.0, 2.0)
        rpm = random.uniform(1000, 1500)

        for i in range(num_records):
            timestamp = start_time + timedelta(seconds=i * 60)
            temperature += np.random.normal(0, 0.05)
            vibration += np.random.normal(0, 0.01)
            pressure += np.random.normal(0, 0.02)
            rpm += np.random.normal(0, 5)

            # Simulate an increasing chance of failure towards the end of each machine's life
            horizon = i / num_records
            local_fail_prob = anomaly_chance + 0.03 * horizon**3
            failure = 1 if random.random() < local_fail_prob else 0

            data.append([timestamp.isoformat(), machine_id, round(temperature, 3),
                         round(vibration, 4), round(pressure, 3), int(rpm), failure])

    df = pd.DataFrame(data, columns=['timestamp', 'machine_id', 'temperature', 'vibration', 'pressure', 'rpm', 'failure'])
    return df

if __name__ == "__main__":
    df = generate_sensor_data(num_machines=8, num_records=1200)
    out_path = OUT / "simulated_sensor_data.csv"
    df.to_csv(out_path, index=False)
    print("✅ Simulated sensor data generated:", out_path)
    print(df.head())
