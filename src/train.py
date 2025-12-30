
import pandas as pd
import warnings
import argparse
import utils 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error
)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings('ignore')

# --- 1. classification ---
def run_classification_pipeline(X_train, y_train, X_test, y_test):
    """
    Trains, evaluates, and saves classification models.
    """
    print("\n--- 🚀 Starting Classification Pipeline ---")
    
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, solver='liblinear'),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss')
    }
    
    metrics_report = {}
    best_model = None
    best_f1 = 0.0
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # evaluate
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision_macro": precision_score(y_test, y_pred, average='macro'),
            "recall_macro": recall_score(y_test, y_pred, average='macro'),
            "f1_macro": f1_score(y_test, y_pred, average='macro')
        }
        
        print(f"Metrics for {name}: {metrics}")
        metrics_report[name] = metrics
        
        # tract the best model
        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_model = model

    # save metrics
    utils.save_metrics(metrics_report, "classification_metrics.json")
    
    # save the best model
    utils.save_model(best_model, "best_classification_model.joblib")
    
    print("--- ✅ Classification Pipeline Finished ---")
    return best_model


# --- 2. regression pipeline ---
def run_regression_pipeline(X_train, y_train_dict, X_test, y_test_dict):
    """
    Trains, evaluates, and saves regression models for each target.
    """
    print("\n--- 🚀 Starting Regression Pipeline ---")
    
    models_to_test = {
        "DecisionTree": DecisionTreeRegressor(random_state=42),
        "RandomForest": RandomForestRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42)
    }
    
    # regression target
    regression_targets = y_train_dict.columns
    metrics_report = {}
    best_models = {}

    # train model for different targets
    for target in regression_targets:
        print(f"\n--- Training for Target: {target} ---")
        y_train = y_train_dict[target]
        y_test = y_test_dict[target]
        metrics_report[target] = {}
        
        best_model_for_target = None
        best_rmse = float('inf')

        # find all the model
        for name, model in models_to_test.items():
            print(f"Training {name} for {target}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # evaluate
            metrics = {
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
            }
            
            print(f"Metrics for {name} ({target}): {metrics}")
            metrics_report[target][name] = metrics
            
            # track the best model
            if metrics["rmse"] < best_rmse:
                best_rmse = metrics["rmse"]
                best_model_for_target = model
        
        # save the model
        utils.save_model(best_model_for_target, f"best_regression_model_{target}.joblib")
        best_models[target] = best_model_for_target

    # store all the regression metrics
    utils.save_metrics(metrics_report, "regression_metrics.json")
    
    print("--- ✅ Regression Pipeline Finished ---")
    return best_models


# --- 3. explanation ---
def generate_explanations(clf_model, reg_models, X_data, X_data_sampled):
    """
    Generates and saves SHAP plots for the best models.
    """
    print("\n--- 🚀 Starting Explainability Pipeline ---")
    
    # 1. classification explain
    try:
        utils.save_shap_plot(clf_model, X_data_sampled, "shap_plot_classification.png")
    except Exception as e:
        print(f"Error generating SHAP for classification model: {e}")

    # 2. regression explain
    try:
        model_to_explain = reg_models.get('total_fatalities')
        if model_to_explain:
            utils.save_shap_plot(model_to_explain, X_data_sampled, "shap_plot_regression_fatalities.png")
    except Exception as e:
        print(f"Error generating SHAP for regression model: {e}")
        
    print("--- ✅ Explainability Pipeline Finished ---")

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--reports_dir", type=str, default=None)

    return parser.parse_args()

# --- 4.  (Main Function) ---
def main():
    """
    Orchestrates the full ML training pipeline.
    """
    print("===== 🚀 STARTING FULL ML PIPELINE =====")
    
    # 1. load data

    args = parse_args()
    utils.init_paths(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
    )
    X, y = utils.load_data()
    if X is None:
        return  
    # 2. load feature
    X_features = X.drop(columns=['event_id', 'GEOID'], errors='ignore')

    # 3. load target
    y_class = y['impact_rank']
    # 将 "low", "medium", "high" 转换为 0, 1, 2
    le = LabelEncoder()
    y_class_encoded = le.fit_transform(y_class)
    print("Saving LabelEncoder...")
    utils.save_model(le, "label_encoder.joblib")


    y_reg_cols = [
        'total_population_affected', 'total_fatalities', 
        'total_injuries', 'total_socio_economic_loss'
    ]
    y_reg = y[y_reg_cols]

    # 4. divide trainning/testing
    X_train, X_test, y_train_class, y_test_class, y_train_reg, y_test_reg = train_test_split(
        X_features, y_class_encoded, y_reg, test_size=0.2, random_state=42
    )
    
    # 5. run-classification
    best_clf = run_classification_pipeline(X_train, y_train_class, X_test, y_test_class)
    
    # 6. run-regression
    best_reg_models = run_regression_pipeline(X_train, y_train_reg, X_test, y_test_reg)
    
    # 7. SHAP
    X_train_sampled = X_train.sample(n=min(1000, len(X_train)), random_state=42)
    generate_explanations(best_clf, best_reg_models, X_train, X_train_sampled)

    print("===== ✅ FULL ML PIPELINE FINISHED SUCCESSFULLY =====")
if __name__ == "__main__":
    main()