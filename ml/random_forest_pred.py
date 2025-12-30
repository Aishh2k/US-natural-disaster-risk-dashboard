import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import os

# ==========================================
# 1.  ()
# ==========================================
def load_and_merge_data(data_folder='./processed_data/'):
    print("...")
    all_merged_data = []
    #  2000-2024 
    years = range(2000, 2025) 
    
    for year in years:
        feat_file = os.path.join(data_folder, f"feature_{year}.csv")
        targ_file = os.path.join(data_folder, f"target_{year}.csv")
        
        if os.path.exists(feat_file) and os.path.exists(targ_file):
            df_feat = pd.read_csv(feat_file)
            df_targ = pd.read_csv(targ_file)
            
            # 
            df_feat.columns = [c.strip() for c in df_feat.columns]
            df_targ.columns = [c.strip() for c in df_targ.columns]
            
            # 
            merge_keys = ['state_abbr', 'month', 'disaster_name']
            
            # Inner Join
            merged_df = pd.merge(df_feat, df_targ, on=merge_keys, how='inner')
            
            if 'year' not in merged_df.columns:
                merged_df['year'] = year
            
            all_merged_data.append(merged_df)
    
    if not all_merged_data:
        print("，！")
        return None 
        
    full_df = pd.concat(all_merged_data, ignore_index=True)
    return full_df

# ==========================================
# 2.  ()
# ==========================================
def visualize_prediction_gap(y_true, y_pred, title_suffix=""):
    """
    y_true:  ( Log )
    y_pred:  ( Log )
    """
    y_pred = np.maximum(y_pred, 1e-6) 
    y_true = np.maximum(y_true, 1e-6)
    
    plt.figure(figsize=(14, 6))
    
    #  1:  vs 
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5, color='green', s=20) # RF
    
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('True Value (Log Scale)')
    plt.ylabel('Predicted Value (Log Scale)')
    plt.title(f'Random Forest: True vs Predicted {title_suffix}')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    #  2: 
    plt.subplot(1, 2, 2)
    subset_size = min(50, len(y_true))
    indices = np.arange(subset_size)
    
    plt.plot(indices, y_true[:subset_size], 'o-', label='True', color='blue', alpha=0.7)
    plt.plot(indices, y_pred[:subset_size], 'x--', label='Predicted', color='orange', alpha=0.7)
    
    plt.yscale('log')
    plt.xlabel('Sample Index')
    plt.ylabel('Value (Log Scale)')
    plt.title(f'Prediction Gap (First {subset_size} Samples)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('output/random_forest_'+f'rf_prediction_analysis_{title_suffix}.png')
    plt.show()
    print(f" random_forest_rf_prediction_analysis_{title_suffix}.png")

# ==========================================
# 3. Random Forest  ()
# ==========================================
def train_one_target_rf(X, y, y_raw, target_name="Loss"):
    
    # 3. Train-test split
    X_train, X_test, y_train, y_test, y_raw_train, y_raw_test = train_test_split(
        X, y, y_raw, test_size=0.2, random_state=42
    )

    # 4. Random Forest Model 
    # ：RF  'early_stopping'， warm_start 
    max_trees = 100  # 
    model = RandomForestRegressor(
        n_estimators=1,        #  1 
        warm_start=True,       # 
        max_depth=10,          #  (RF)
        n_jobs=-1,
        random_state=42,
        criterion='squared_error' # RF  tweedie， MSE
    )

    print(f"\n========================")
    print(f": {target_name}")
    print(f"========================")

    train_rmse = []
    val_rmse = []
    
    # 5.  ()
    #  XGBoost  Loss 
    for i in range(1, max_trees + 1):
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)
        
        #  Loss (RMSE)
        # ：RandomForest  Loss （）
        # 
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_test)
        
        t_loss = np.sqrt(mean_squared_error(y_train, train_pred))
        v_loss = np.sqrt(mean_squared_error(y_test, val_pred))
        
        train_rmse.append(t_loss)
        val_rmse.append(v_loss)
        
        if i % 10 == 0:
            print(f"Trees: {i}/{max_trees} | Train RMSE: {t_loss:.4f} | Val RMSE: {v_loss:.4f}")

    # 6. Plot loss curve (Error vs Number of Trees)
    x_axis = range(1, max_trees + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(x_axis, train_rmse, label='Train RMSE')
    plt.plot(x_axis, val_rmse, label='Validation RMSE')
    plt.legend()
    plt.xlabel('Number of Trees')
    plt.ylabel('RMSE (Log Scale Data)')
    plt.title(f'Random Forest Learning Curve - {target_name}')
    plt.grid(True)
    fname = 'output/random_forest_'+f"rf_loss_curve_{target_name}.png"
    plt.savefig(fname)
    plt.close()
    print(f"[OK] Loss  {fname}")

    # 8. Predictions
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    
    # --- ： Loss  log 
    if target_name == "Loss":
        preds_real = np.expm1(preds)
        y_test_real = y_raw_test
    else:
        preds_real = preds #  Fatality  Log 
        #  Fatality  log1p， expm1
        if np.max(y_test) < 100: #  Log 
             preds_real = np.expm1(preds)
        y_test_real = y_raw_test

    preds_real = np.maximum(preds_real, 0)  # 

    # --- 
    visualize_prediction_gap(y_test_real, preds_real, title_suffix=target_name)

    print(f"\n--- {target_name}  vs  (10) ---")
    print(pd.DataFrame({"True": y_test_real, "Pred": preds_real}).head(10))
    
    print(f"[{target_name}]  MSE(): {mean_squared_error(y_test_real, preds_real):.4f}")

    return model

# ==========================================
# 4. 
# ==========================================
def train_and_visualize_rf():
    data = load_and_merge_data()
    if data is None:
        return

    print(f": {len(data)}")

    # Features
    numerical_features = [
        'year', 'count', 'SOVI_SCORE', 'RESL_SCORE',
        'pop_density', 'avg_eal_valt'
    ]
    categorical_features = ['state_abbr', 'disaster_name']

    data[numerical_features] = data[numerical_features].fillna(0)

    X = pd.get_dummies(
        data[numerical_features + categorical_features],
        columns=categorical_features,
        dtype=float
    )

    # Target 1: avg_loss (Log )
    y_loss_raw = data['avg_loss'].fillna(0)
    y_loss = np.log1p(y_loss_raw)

    # Target 2: avg_fatality ( Log )
    y_fatal_raw = data['avg_fatality'].fillna(0)
    y_fatal = np.log1p(y_fatal_raw)

    print("\n==========  Random Forest ==========")

    # Model A: Loss
    model_loss = train_one_target_rf(X, y_loss, y_loss_raw, target_name="Loss")

    # Model B: Fatality
    model_fatal = train_one_target_rf(X, y_fatal, y_fatal_raw, target_name="Fatality")

    print("\n🎉 Random Forest ！")

if __name__ == "__main__":
    train_and_visualize_rf()