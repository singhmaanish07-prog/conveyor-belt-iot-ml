Automated Conveyor Belt with IoT & Predictive Maintenance
Data Science & AI Project – Mechanical Engineering
-----------------------------------------------------

This single script simulates sensor data, extracts features,
trains a RandomForest model, evaluates it, and runs a 
mock real-time inference loop.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib, json, time


# --------------------------
# 1. Simulate IoT Sensor Data
# --------------------------
def simulate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    start = datetime(2025, 1, 1, 8, 0, 0)
    timestamps = [start + timedelta(seconds=5 * i) for i in range(n)]

    vib_healthy = rng.normal(0.8, 0.15, n)
    temp_healthy = rng.normal(45, 2.5, n)
    speed_healthy = rng.normal(0.9, 0.05, n)
    load_healthy = rng.normal(15, 3.0, n)

    # Fault injection
    fault_mask = rng.random(n) < 0.22
    vib = vib_healthy + fault_mask * rng.normal(0.7, 0.25, n)
    temp = temp_healthy + fault_mask * rng.normal(6.0, 2.0, n)
    speed = speed_healthy - fault_mask * rng.normal(0.15, 0.05, n)
    load = np.clip(load_healthy + fault_mask * rng.normal(2.0, 1.0, n), 5, None)

    # Label: faulty if vibration, temperature, or speed exceed safe thresholds
    label = ((vib > 1.3) | (temp > 52) | (speed < 0.78)).astype(int)

    return pd.DataFrame({
        "timestamp": timestamps,
        "vibration_rms": vib.round(3),
        "temperature_c": temp.round(2),
        "belt_speed_mps": speed.round(3),
        "load_kg": load.round(2),
        "label": label
    })


# --------------------------
# 2. Feature Engineering
# --------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["vib_roll_mean"] = df["vibration_rms"].rolling(12, min_periods=1).mean()
    df["temp_roll_mean"] = df["temperature_c"].rolling(12, min_periods=1).mean()
    df["speed_roll_mean"] = df["belt_speed_mps"].rolling(12, min_periods=1).mean()
    df["vib_diff"] = df["vibration_rms"].diff().fillna(0)
    df["temp_diff"] = df["temperature_c"].diff().fillna(0)
    df["speed_diff"] = df["belt_speed_mps"].diff().fillna(0)
    df["vib_temp_mul"] = df["vibration_rms"] * df["temperature_c"]
    df["load_speed_ratio"] = (df["load_kg"] / (df["belt_speed_mps"] + 1e-6)).clip(0, 1000)
    return df.drop(columns=["timestamp"])


# --------------------------
# 3. Train & Save Model
# --------------------------
def train_model(df):
    Xy = add_features(df)
    y = Xy.pop("label")

    X_train, X_test, y_train, y_test = train_test_split(
        Xy, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=120, max_depth=8,
                                   class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    print("\n=== Model Evaluation ===")
    print("Accuracy :", metrics["accuracy"])
    print("Precision:", metrics["precision"])
    print("Recall   :", metrics["recall"])
    print("Confusion Matrix:", metrics["confusion_matrix"])
    print("\nDetailed Report:\n", classification_report(y_test, y_pred))

    joblib.dump(model, "model.joblib")
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    return model


# --------------------------
# 4. Real-time Inference Demo
# --------------------------
def real_time_demo(model, n=20):
    rng = np.random.default_rng(123)
    print("\n=== Real-time Mock Inference ===")
    for i in range(n):
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "vibration_rms": round(float(rng.normal(0.9, 0.2)), 3),
            "temperature_c": round(float(rng.normal(46, 3.0)), 2),
            "belt_speed_mps": round(float(rng.normal(0.88, 0.07)), 3),
            "load_kg": round(float(rng.normal(15, 3.0)), 2)
        }
        df = pd.DataFrame([row])
        X = add_features(df).drop(columns=["label"], errors="ignore")
        pred = int(model.predict(X)[0])
        status = "ALERT ⚠ Fault risk" if pred == 1 else "OK ✅"
        print({**row, "prediction": status})
        time.sleep(0.1)


# --------------------------
# 5. Run the Pipeline
# --------------------------
if _name_ == "_main_":
    df = simulate_data()
    df.to_csv("sample_data.csv", index=False)  # Save simulated dataset
    model = train_model(df)
    real_time_demo(model, n=10)
