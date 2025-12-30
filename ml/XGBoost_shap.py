import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import os
import joblib
import shap  # <--- NEW:  SHAP 

# ==========================================
# 0.  (Two-Stage Model)
# ==========================================
class TwoStageXGBoost:
    def __init__(self, classification_params=None, regression_params=None):
        #  ( Loss > 0)
        self.clf_params = classification_params if classification_params else {
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
            'objective': 'binary:logistic', 'n_jobs': -1, 'random_state': 42,
            'scale_pos_weight': 5 
        }
        #  ( Loss )
        self.reg_params = regression_params if regression_params else {
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
            'objective': 'reg:squarederror', 'n_jobs': -1, 'random_state': 42
        }
        
        self.classifier = xgb.XGBClassifier(**self.clf_params)
        self.regressor = xgb.XGBRegressor(**self.reg_params)
        self.clf_evals_result = {}
        self.reg_evals_result = {}
        
    def fit(self, X_train, y_train, X_val, y_val):
        # --- 1.  ---
        y_train_bin = (y_train > 0).astype(int)
        y_val_bin = (y_val > 0).astype(int)
        
        print(f"  [TwoStage]  (Class 1 : {y_train_bin.mean():.2%})...")
        self.classifier.fit(
            X_train, y_train_bin,
            eval_set=[(X_train, y_train_bin), (X_val, y_val_bin)],
            eval_metric='logloss',
            verbose=False
        )
        self.clf_evals_result = self.classifier.evals_result()
        
        # --- 2.  ---
        mask_train = y_train > 0
        mask_val = y_val > 0 
        
        if mask_train.sum() > 10:
            print(f"  [TwoStage]  (: {mask_train.sum()})...")
            self.regressor.fit(
                X_train[mask_train], y_train[mask_train],
                eval_set=[(X_train[mask_train], y_train[mask_train]), 
                          (X_val[mask_val], y_val[mask_val])],
                eval_metric='rmse',
                verbose=False
            )
            self.reg_evals_result = self.regressor.evals_result()
        else:
            print("  [TwoStage] ，！")
            self.regressor = None
            
    def predict(self, X):
        prob_exists = self.classifier.predict_proba(X)[:, 1]
        is_existing = (prob_exists > 0.5).astype(int)
        
        if self.regressor:
            pred_values = self.regressor.predict(X)
            pred_values = np.maximum(pred_values, 0)
        else:
            pred_values = np.zeros(len(X))
            
        return is_existing * pred_values

# ==========================================
# 1.  ()
# ==========================================
def visualize_prediction_gap(y_true, y_pred, title_suffix=""):
    y_pred = np.maximum(y_pred, 1e-6)
    y_true = np.maximum(y_true, 1e-6)
    
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, color='blue', s=20)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Value (Log Scale)')
    plt.ylabel('Predicted Value (Log Scale)')
    plt.title(f'True vs Predicted {title_suffix}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.subplot(1, 2, 2)
    subset_size = min(50, len(y_true))
    indices = np.arange(subset_size)
    plt.plot(indices, y_true[:subset_size], 'o-', label='True', color='green', alpha=0.7)
    plt.plot(indices, y_pred[:subset_size], 'x--', label='Predicted', color='red', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Sample Index')
    plt.ylabel('Value (Log Scale)')
    plt.title(f'Prediction Gap (First {subset_size} Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = f'output/prediction_analysis_{title_suffix}.png'
    plt.savefig(fname)
    plt.close()
    print(f" : {fname}")

# ==========================================
# NEW: SHAP  ()
# ==========================================
def run_shap_analysis(model, X, target_name="Loss"):
    """
     Two-Stage  SHAP 。
    """
    print(f"\n🔎 Running SHAP Analysis for {target_name}...")
    
    # 1.  (Classifier Explainability)
    # TreeExplainer  XGBoost
    explainer_clf = shap.TreeExplainer(model.classifier)
    shap_values_clf = explainer_clf.shap_values(X)
    
    plt.figure(figsize=(10, 6))
    plt.title(f"SHAP Summary - {target_name} Classifier (Probability)")
    # summary_plot 
    shap.summary_plot(shap_values_clf, X, show=False)
    plt.tight_layout()
    plt.savefig(f'output/shap_summary_{target_name}_classifier.png')
    plt.close()
    print(f"   -> Saved Classifier SHAP plot")

    # 2.  (Regressor Explainability)
    if model.regressor:
        explainer_reg = shap.TreeExplainer(model.regressor)
        shap_values_reg = explainer_reg.shap_values(X)
        
        plt.figure(figsize=(10, 6))
        plt.title(f"SHAP Summary - {target_name} Regressor (Magnitude)")
        shap.summary_plot(shap_values_reg, X, show=False)
        plt.tight_layout()
        plt.savefig(f'output/shap_summary_{target_name}_regressor.png')
        plt.close()
        print(f"   -> Saved Regressor SHAP plot")
    else:
        print("   -> Regressor not trained, skipping SHAP.")

# ==========================================
# 2.  (Single Target Two-Stage)
# ==========================================
def train_one_target(X, y, y_raw, target_name="Loss"):
    # Train-test split
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=42
    )

    print(f"\n========================")
    print(f": {target_name}")
    print(f"========================")

    scale_weight = 10 if target_name == "Fatality" else 5
    
    model = TwoStageXGBoost(
        classification_params={
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
            'objective': 'binary:logistic', 'scale_pos_weight': scale_weight,
            'n_jobs': -1, 'random_state': 42
        },
        regression_params={
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
            'objective': 'reg:squarederror',
            'n_jobs': -1, 'random_state': 42
        }
    )

    model.fit(X_train, y_train, X_test, y_test)

    # ---  Loss  () ---
    if model.clf_evals_result:
        res = model.clf_evals_result
        epochs = len(res['validation_0']['logloss'])
        x_axis = range(0, epochs)
        plt.figure(figsize=(8, 4))
        plt.plot(x_axis, res['validation_0']['logloss'], label='Train')
        plt.plot(x_axis, res['validation_1']['logloss'], label='Val')
        plt.title(f'{target_name} Classifier Loss (LogLoss)')
        plt.legend()
        plt.savefig('output/'+f'loss_curve_{target_name}_classifier.png')
        plt.close()
        
    if model.reg_evals_result:
        res = model.reg_evals_result
        if 'validation_0' in res and 'rmse' in res['validation_0']:
            epochs = len(res['validation_0']['rmse'])
            x_axis = range(0, epochs)
            plt.figure(figsize=(8, 4))
            plt.plot(x_axis, res['validation_0']['rmse'], label='Train')
            if 'validation_1' in res:
                plt.plot(x_axis, res['validation_1']['rmse'], label='Val')
            plt.title(f'{target_name} Regressor Loss (RMSE)')
            plt.legend()
            plt.savefig('output/'+f'loss_curve_{target_name}_regressor.png')
            plt.close()

    print(f"[OK] ，...")

    # ---  ---
    preds = model.predict(X_test)
    
    if target_name == "Loss":
        preds_real = np.expm1(preds)
        y_test_real = y_raw_test
    else:
        preds_real = np.expm1(preds)
        y_test_real = y_raw_test

    preds_real = np.maximum(preds_real, 0)

    mse = mean_squared_error(y_test_real, preds_real)
    
    y_bin_true = (y_test_real > 0).astype(int)
    y_bin_pred = (preds_real > 0).astype(int)
    acc = accuracy_score(y_bin_true, y_bin_pred)

    visualize_prediction_gap(y_test_real, preds_real, title_suffix=target_name)

    print(f"\n--- {target_name}  vs  (10) ---")
    df_res = pd.DataFrame({"True": y_test_real, "Pred": preds_real})
    print(df_res.head(10))
    
    print(f"[{target_name}]  MSE(): {mse:.4f}")
    print(f"[{target_name}] : {acc:.2%}")
    
    # ==========================================
    # <--- NEW:  SHAP 
    #  X_test ，
    # ==========================================
    run_shap_analysis(model, X_test, target_name=target_name)

    model_filename = 'output_model/'+ f"two_stage_xgb_{target_name}.pkl"
    joblib.dump(model, model_filename)
    print(f"✅ : {model_filename}")
    return model

# ==========================================
# 3.  ()
# ==========================================
def load_and_merge_data(data_folder='./processed_data/'):
    print("...")
    all_merged_data = []
    years = range(2000, 2025) 
    
    for year in years:
        feat_file = os.path.join(data_folder, f"feature_{year}.csv")
        targ_file = os.path.join(data_folder, f"target_{year}.csv")
        
        if os.path.exists(feat_file) and os.path.exists(targ_file):
            df_feat = pd.read_csv(feat_file)
            df_targ = pd.read_csv(targ_file)
            
            df_feat.columns = [c.strip() for c in df_feat.columns]
            df_targ.columns = [c.strip() for c in df_targ.columns]
            
            merge_keys = ['state_abbr', 'month', 'disaster_name']
            merged_df = pd.merge(df_feat, df_targ, on=merge_keys, how='inner')
            
            if 'year' not in merged_df.columns:
                merged_df['year'] = year
            
            all_merged_data.append(merged_df)
    
    if not all_merged_data:
        print("！")
        return None 
        
    full_df = pd.concat(all_merged_data, ignore_index=True)
    return full_df

# ==========================================
# 4.  ()
# ==========================================
def train_and_visualize():
    data = load_and_merge_data()
    if data is None: return

    print(f": {len(data)}")

    # Features ()
    data['pop_x_count'] = data['pop_density'] * data['count']
    data['sovi_x_count'] = data['SOVI_SCORE'] * data['count']
    
    numerical_features = [
        'year', 'count', 'SOVI_SCORE', 'RESL_SCORE',
        'pop_density', 'avg_eal_valt',
        'pop_x_count', 'sovi_x_count'
    ]
    categorical_features = ['state_abbr', 'disaster_name']

    data[numerical_features] = data[numerical_features].fillna(0)

    X = pd.get_dummies(
        data[numerical_features + categorical_features],
        columns=categorical_features,
        dtype=float
    )

    # Target 1: Loss (Log)
    y_loss_raw = data['avg_loss'].fillna(0)
    y_loss = np.log1p(y_loss_raw)

    # Target 2: Fatality (Log)
    y_fatal_raw = data['avg_fatality'].fillna(0)
    y_fatal = np.log1p(y_fatal_raw)

    print("\n==========  ==========")

    # Model A: Loss
    model_loss = train_one_target(X, y_loss, y_loss_raw, target_name="Loss")

    # Model B: Fatality
    model_fatal = train_one_target(X, y_fatal, y_fatal_raw, target_name="Fatality")

    print("\n🎉 Saved！")

if __name__ == "__main__":
    train_and_visualize()