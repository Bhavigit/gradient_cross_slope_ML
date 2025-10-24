"""
Gradient and Crossfall Prediction Pipeline
------------------------------------------
This script processes sensor-based JSON data, aggregates features per window,
trains Random Forest models for gradient and crossfall prediction, and
generates CSV predictions and plots.

"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------- CONFIG --------------------------
DATA_DIR = r"C:\Users\KarriBhavya\PycharmProjects\gradient_cross_slope_ML\Data"
GROUND_TRUTH_FILE = os.path.join(DATA_DIR, "geo_labels(in).csv")

JSON_SECTION_MAP = {
    "1000~~Test-Videos 2023-2024~7674~Video_1752485838.json": None,
    "1000~Lonrix Test Network~Lonrix Test Videos~7674~Video_1754971873.json": None,
    "1002~Hamilton~Lonrix Test Videos~115~Video_1756692526.json": None,
    "10009~Juno App WW Geometry Test~Lonrix Test Videos~4518~video_7175345834.json": None
}

OUTPUT_RESULTS_DIR = "results_ml1_final"
OUTPUT_PLOTS_DIR = "plots_ml1_final"
WINDOW_SIZE = 50# number of JSON records per segment

os.makedirs(OUTPUT_RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)


# ------------------------ FUNCTIONS ------------------------
def load_ground_truth(csv_path):
    """
    Load ground truth CSV containing section-wise gradient and crossfall.

    Args:
        csv_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Ground truth dataframe with numeric gradients and crossfall
    """
    print("Loading ground truth...")
    gt = pd.read_csv(csv_path)
    gt = gt.replace("00:00.0", 0)
    gt["gradients"] = pd.to_numeric(gt["gradients"], errors="coerce")
    gt["crossfall"] = pd.to_numeric(gt["crossfall"], errors="coerce")
    gt.dropna(subset=["gradients", "crossfall"], inplace=True)
    gt["sectionID"] = range(1, len(gt) + 1)
    print(f"Loaded {len(gt)} rows of ground truth.")
    return gt


def load_json_data(filepath):
    """
    Load sensor JSON data and extract mean accelerometer and gyro values per record.

    Args:
        filepath (str): Path to JSON file

    Returns:
        pd.DataFrame: Dataframe with features per record
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for rec in data:
        accel = rec.get("accelerometer", [])
        gyro = rec.get("gyroscope", [])
        if accel:
            records.append({
                "lat": rec.get("lat"),
                "lon": rec.get("lon"),
                "speed": float(rec.get("speed", 0)),
                "accel_x": np.mean([a["x"] for a in accel]),
                "accel_y": np.mean([a["y"] for a in accel]),
                "accel_z": np.mean([a["z"] for a in accel]),
                "gyro_z": np.mean([g["z"] for g in gyro]) if gyro else 0.0
            })
    return pd.DataFrame(records)


def aggregate_features(df, start_section_id, window_size=WINDOW_SIZE):
    """
    Aggregate features over sliding windows and assign sequential section IDs.

    Args:
        df (pd.DataFrame): Raw JSON features
        start_section_id (int): Starting sectionID for this dataset
        window_size (int): Number of records per segment

    Returns:
        pd.DataFrame: Aggregated features with sectionID
    """
    n_windows = ceil(len(df) / window_size)
    aggregated_list = []

    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window = df.iloc[start:end]
        if window.empty:
            continue
        agg = window.agg({
            "accel_x": "mean",
            "accel_y": "mean",
            "accel_z": "mean",
            "gyro_z": "mean",
            "speed": "mean",
            "lat": "mean",
            "lon": "mean"
        }).to_frame().T
        agg["sectionID"] = start_section_id + i
        aggregated_list.append(agg)

    return pd.concat(aggregated_list, ignore_index=True)


def train_models(gt_df, features_df):
    """
    Train Random Forest models for gradient and crossfall prediction.

    Args:
        gt_df (pd.DataFrame): Ground truth
        features_df (pd.DataFrame): Aggregated feature dataframe

    Returns:
        tuple: (gradient_model, crossfall_model)
    """
    feature_cols = ["accel_x", "accel_y", "accel_z", "gyro_z", "speed"]
    merged = pd.merge(features_df, gt_df, on="sectionID", how="inner")
    print(f"Training models with {len(merged)} merged samples.")

    if len(merged) < 5:
        print("Not enough samples for training.")
        return None, None

    X = merged[feature_cols]
    y_grad = merged["gradients"]
    y_cross = merged["crossfall"]

    X_train, X_test, y_grad_train, y_grad_test, y_cross_train, y_cross_test = train_test_split(
        X, y_grad, y_cross, test_size=0.3, random_state=42
    )

    model_grad = RandomForestRegressor(n_estimators=100, random_state=42)
    model_cross = RandomForestRegressor(n_estimators=100, random_state=42)

    model_grad.fit(X_train, y_grad_train)
    model_cross.fit(X_train, y_cross_train)

    print(f"Gradient MSE: {mean_squared_error(y_grad_test, model_grad.predict(X_test)):.3f}")
    print(f"Crossfall MSE: {mean_squared_error(y_cross_test, model_cross.predict(X_test)):.3f}")

    return model_grad, model_cross


def predict_and_plot(model_grad, model_cross, features_df, gt_df, output_csv, prefix):
    """
    Make predictions and generate comparison plots for gradient and crossfall.

    Args:
        model_grad: Trained gradient model
        model_cross: Trained crossfall model
        features_df (pd.DataFrame): Aggregated features
        gt_df (pd.DataFrame): Ground truth
        output_csv (str): Path to save predictions
        prefix (str): Plot title prefix
    """
    feature_cols = ["accel_x", "accel_y", "accel_z", "gyro_z", "speed"]
    merged = pd.merge(features_df, gt_df, on="sectionID", how="left")
    X = merged[feature_cols].fillna(0)

    merged["pred_gradients"] = model_grad.predict(X)
    merged["pred_crossfall"] = model_cross.predict(X)
    merged.to_csv(output_csv, index=False)
    print(f"Saved predictions to {output_csv}")

    # Gradient plot
    plt.figure(figsize=(12,5))
    plt.plot(merged["pred_gradients"], label="Predicted Gradient", color="blue")
    plt.plot(merged["gradients"], label="True Gradient", color="red", linestyle="--")
    plt.xlabel("Section ID")
    plt.ylabel("Gradient (%)")
    plt.title(f"{prefix} - Gradient Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f"{prefix}_gradient_plot.png"), dpi=300)
    plt.close()

    # Crossfall plot
    plt.figure(figsize=(12,5))
    plt.plot(merged["pred_crossfall"], label="Predicted Crossfall", color="green")
    plt.plot(merged["crossfall"], label="True Crossfall", color="orange", linestyle="--")
    plt.xlabel("Section ID")
    plt.ylabel("Crossfall (%)")
    plt.title(f"{prefix} - Crossfall Prediction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_PLOTS_DIR, f"{prefix}_crossfall_plot.png"), dpi=300)
    plt.close()


# -------------------------- MAIN PIPELINE --------------------------
def main():
    print("Starting ML Pipeline...\n")

    # Load ground truth
    gt_df = load_ground_truth(GROUND_TRUTH_FILE)

    feature_list = []
    section_start = 1

    # Process each JSON file
    for fname in JSON_SECTION_MAP.keys():
        fpath = os.path.join(DATA_DIR, fname)
        print(f"\nProcessing file: {fname}")

        df = load_json_data(fpath)
        if df.empty:
            print("No valid sensor data found, skipping.")
            continue

        agg_df = aggregate_features(df, start_section_id=section_start)
        section_start += len(agg_df)
        feature_list.append(agg_df)

    if not feature_list:
        print("No JSON features loaded. Exiting.")
        return

    combined_features = pd.concat(feature_list, ignore_index=True)

    # Train models
    model_grad, model_cross = train_models(gt_df, combined_features)
    if not model_grad:
        print("Training failed. Exiting.")
        return

    # Generate predictions and plots
    for fname in JSON_SECTION_MAP.keys():
        fpath = os.path.join(DATA_DIR, fname)
        base = os.path.splitext(os.path.basename(fname))[0]
        df = load_json_data(fpath)
        agg_df = aggregate_features(df, start_section_id=1)
        out_csv = os.path.join(OUTPUT_RESULTS_DIR, f"{base}_pred.csv")
        predict_and_plot(model_grad, model_cross, agg_df, gt_df, out_csv, base)

    print("\nPipeline finished successfully!")


if __name__ == "__main__":
    main()
