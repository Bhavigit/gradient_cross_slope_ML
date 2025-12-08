"""
Gradient & Crossfall Prediction using Random Forest
-------------------------------------------------
Removes mediaTime & frameNumber
Outputs per-segment predictions
adds GPS from/to (latitude_from, longitude_from, latitude_to, longitude_to)

"""

import os, json, numpy as np, pandas as pd
from math import ceil
from sklearn.ensemble import RandomForestRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR = r"C:\Users\KarriBhavya\PycharmProjects\gradient_cross_slope_ML\data_1"
GROUND_TRUTH_FILE = os.path.join(DATA_DIR, "geometry_truth(in)_1.csv")

JSON_FILES = [
    "10009~Juno App WW Geometry Test~Lonrix Test Videos~4518~video_7175345834.json",
     # "1002~Hamilton~Lonrix Test Videos~115~Video_1756692526.json",
    "10009~Juno App WW Geometry Test~LonrixTestVideos~4518~video_7608442085.json",
    # "10009~Juno App WW Geometry Test~Lonrix Test Videos~4518~video_7175345834 (1).json",
    # "10009~Juno App WW Geometry Test~Lonrix Test Videos~4518~video_7175345834 (2).json"
    "10009~Juno App WW Geometry Test~LonrixTestVideos~4518~video_4944342144.json",
    "10009~Juno App WW Geometry Test~LonrixTestVideos~4518~video_5399854598.json"
]


RESULTS_DIR = "results_1"
DIR_SEGMENTS = os.path.join(RESULTS_DIR, "per_segment")
DIR_METRICS = os.path.join(RESULTS_DIR, "metrics")
DIR_MODELS = os.path.join(RESULTS_DIR, "models")


# Create folders
os.makedirs(DIR_SEGMENTS, exist_ok=True)
os.makedirs(DIR_METRICS, exist_ok=True)
os.makedirs(DIR_MODELS, exist_ok=True)


ROLL_WINDOW_FRAMES = 20
FRAMES_PER_10M = 20


# ---------------- HELPERS ----------------
def load_ground_truth(path):
    gt = pd.read_csv(path).replace("00:00.0", 0)
    gt["gradient"] = pd.to_numeric(gt["gradient"], errors="coerce")
    gt["crossfall"] = pd.to_numeric(gt["crossfall"], errors="coerce")
    gt.dropna(subset=["gradient", "crossfall"], inplace=True)
    gt.reset_index(drop=True, inplace=True)
    return gt[["gradient", "crossfall"]]


def load_json_rows(fpath):
    data = json.load(open(fpath, encoding="utf-8"))
    rows = []
    for rec in data:
        acc = rec.get("accelerometer", [])
        if not acc:
            continue

        rows.append({
            "lat": rec.get("lat"),
            "lon": rec.get("lon"),
            "speed": float(rec.get("speed", 0)),
            "ax": np.mean([a["x"] for a in acc]),
            "ay": np.mean([a["y"] for a in acc]),
            "az": np.mean([a["z"] for a in acc]),
            "gz": np.mean([g["z"] for g in rec.get("gyroscope", [])]) if rec.get("gyroscope") else 0
        })

    return pd.DataFrame(rows)


# ---------------- UPDATED AGGREGATION ----------------
def aggregate_to_10m_windows(df, start_section=1, frames_per_window=FRAMES_PER_10M):
    out = []
    n = ceil(len(df) / frames_per_window)

    for i in range(n):
        w = df.iloc[i * frames_per_window:(i + 1) * frames_per_window]
        if w.empty:
            continue

        agg = w[["ax", "ay", "az", "gz", "speed"]].mean().to_frame().T

        # GPS FROM–TO
        agg["latitude_from"] = w.iloc[0]["lat"]
        agg["longitude_from"] = w.iloc[0]["lon"]
        agg["latitude_to"] = w.iloc[-1]["lat"]
        agg["longitude_to"] = w.iloc[-1]["lon"]

        agg["sectionID"] = start_section + i
        out.append(agg)

    return pd.concat(out, ignore_index=True)


# ---------------- MODEL TRAINING ----------------
def train_rf(gt, feat):
    merged = feat.merge(gt, left_on="sectionID", right_index=True, how="inner")

    X = merged[["ax", "ay", "az", "gz", "speed"]]
    y_grad = merged["gradient"]
    y_cross = merged["crossfall"]

    # Train-test split
    Xtr_g, Xte_g, ytr_g, yte_g = train_test_split(X, y_grad, test_size=0.3, random_state=42)
    Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, y_cross, test_size=0.3, random_state=42)

    # Models
    rf_g = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_c = RandomForestRegressor(n_estimators=300, random_state=42)
    rf_g.fit(Xtr_g, ytr_g)
    rf_c.fit(Xtr_c, ytr_c)

    # Baseline
    dum_g = DummyRegressor(strategy="mean").fit(Xtr_g, ytr_g)
    dum_c = DummyRegressor(strategy="mean").fit(Xtr_c, ytr_c)

    # Predictions
    pg = rf_g.predict(Xte_g)
    pc = rf_c.predict(Xte_c)
    dg = dum_g.predict(Xte_g)
    dc = dum_c.predict(Xte_c)

    # Metrics
    metrics_path = os.path.join(DIR_METRICS, "metrics_summary.csv")
    pd.DataFrame([{
        "gradient_rmse": np.sqrt(mean_squared_error(yte_g, pg)),
        "crossfall_rmse": np.sqrt(mean_squared_error(yte_c, pc)),
        "gradient_dummy_rmse": np.sqrt(mean_squared_error(yte_g, dg)),
        "crossfall_dummy_rmse": np.sqrt(mean_squared_error(yte_c, dc)),
        "gradient_corr": np.corrcoef(yte_g, pg)[0, 1],
        "crossfall_corr": np.corrcoef(yte_c, pc)[0, 1]
    }]).to_csv(metrics_path, index=False)

    # Save models
    joblib.dump(rf_g, os.path.join(DIR_MODELS, "rf_gradient.joblib"))
    joblib.dump(rf_c, os.path.join(DIR_MODELS, "rf_crossfall.joblib"))

    return rf_g, rf_c


# ---------------- SEGMENT-LEVEL PREDICTION ----------------
def predict_per_segment_csv(rf_g, rf_c, df_rows, out_csv):
    if df_rows is None or len(df_rows) == 0:
        return

    results = []
    n_segments = ceil(len(df_rows) / FRAMES_PER_10M)

    for i in range(n_segments):
        w = df_rows.iloc[i * FRAMES_PER_10M:(i + 1) * FRAMES_PER_10M]
        if w.empty:
            continue

        feats = w[["ax", "ay", "az", "gz", "speed"]].mean().to_frame().T

        pred_g = rf_g.predict(feats)[0]
        pred_c = rf_c.predict(feats)[0]

        lat_from = w.iloc[0]["lat"]
        lon_from = w.iloc[0]["lon"]
        lat_to = w.iloc[-1]["lat"]
        lon_to = w.iloc[-1]["lon"]

        results.append({
            "longitude_from": lon_from,
            "latitude_from": lat_from,
            "longitude_to": lon_to,
            "latitude_to": lat_to,
            "predicted_gradient": pred_g,
            "predicted_crossfall": pred_c
        })

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print("Saved per-segment predictions →", out_csv)


# ---------------- MAIN ----------------
def main():
    gt = load_ground_truth(GROUND_TRUTH_FILE)

    all_feats = []
    section_start = 1

    for fname in JSON_FILES:
        print("Processing:", fname)
        df_rows = load_json_rows(os.path.join(DATA_DIR, fname))
        agg = aggregate_to_10m_windows(df_rows, section_start)
        all_feats.append(agg)
        section_start += len(agg)

    all_feats = pd.concat(all_feats, ignore_index=True)

    rf_g, rf_c = train_rf(gt, all_feats)

    for fname in JSON_FILES:
        base = os.path.splitext(fname)[0]
        df_rows = load_json_rows(os.path.join(DATA_DIR, fname))
        out_path = os.path.join(DIR_SEGMENTS, f"{base}_segments_rf.csv")
        predict_per_segment_csv(rf_g, rf_c, df_rows, out_path)


if __name__ == "__main__":
    main()
