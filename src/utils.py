

import pandas as pd
import joblib
import json
import shap
import matplotlib.pyplot as plt
import os


def init_paths(data_dir=None, model_dir=None, reports_dir=None):
    """
    Initialize directory paths from external args.
    """
    global DATA_DIR, MODEL_DIR, METRICS_DIR, FIGURES_DIR


    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = data_dir or os.path.join(base_dir, 'data', 'input')
    MODEL_DIR = model_dir or os.path.join(base_dir, 'models')

    reports_base = reports_dir or os.path.join(base_dir, 'reports')
    METRICS_DIR = os.path.join(reports_base, 'metrics')
    FIGURES_DIR = os.path.join(reports_base, 'figures')

    # Make sure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(METRICS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("[utils] Paths initialized:")
    print(" DATA_DIR:", DATA_DIR)
    print(" MODEL_DIR:", MODEL_DIR)
    print(" METRICS_DIR:", METRICS_DIR)
    print(" FIGURES_DIR:", FIGURES_DIR)

def load_object(filename):
    """
    Loads a Python object from the /models directory.
    """
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: Object file not found at {path}")
        return None
    
    print(f"Loading object from {path}...")
    return joblib.load(path)

# --- 2. (Data Loading) ---
def load_data():
    """
    Loads features and targets based on the data contract.
    """
    print("Loading data...")
    try:
        features_path = os.path.join(DATA_DIR, 'features.csv')
        targets_path = os.path.join(DATA_DIR, 'targets.csv')
        
        X = pd.read_csv(features_path)
        y = pd.read_csv(targets_path)
        
        print("Data loaded successfully.")
        return X, y
    except FileNotFoundError:
        print(f"Error: Could not find data files in {DATA_DIR}.")
        print("Please ensure 'features.csv' and 'targets.csv' exist.")
        return None, None

# --- 3.  (Model Saving) ---
def save_model(model, filename):
    """
    Saves a trained model to the /models directory.
    """
    path = os.path.join(MODEL_DIR, filename)
    print(f"Saving model to {path}...")
    joblib.dump(model, path)
    print("Model saved.")

# --- 4. (Report Saving) ---
def save_metrics(metrics_dict, filename):
    """
    Saves a dictionary of metrics to a JSON file in /reports/metrics.
    """
    path = os.path.join(METRICS_DIR, filename)
    print(f"Saving metrics to {path}...")
    with open(path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print("Metrics saved.")

def save_shap_plot(model, X_data, filename):
    """
    Generates, saves, and closes a SHAP summary plot.
    """
    print(f"Generating SHAP plot for {filename}...")
    path = os.path.join(FIGURES_DIR, filename)
    
    explainer = shap.TreeExplainer(model, X_data)
    shap_values = explainer(X_data)
    
    plt.figure()
    shap.summary_plot(shap_values, X_data, show=False, plot_type="dot")
    plt.title(f"SHAP Summary - {filename.split('.')[0]}")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    print(f"SHAP plot saved to {path}.")