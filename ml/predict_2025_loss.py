import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os

# ==========================================
# 1.  TwoStageXGBoost 
# ==========================================
#  joblib ，
class TwoStageXGBoost:
    def __init__(self, classification_params=None, regression_params=None):
        self.clf_params = classification_params
        self.reg_params = regression_params
        self.classifier = xgb.XGBClassifier(**self.clf_params) if self.clf_params else None
        self.regressor = xgb.XGBRegressor(**self.reg_params) if self.reg_params else None

    def fit(self, X, y): pass
    
    def predict(self, X):
        # Step 1: 
        prob_exists = self.classifier.predict_proba(X)[:, 1]
        is_existing = (prob_exists > 0.5).astype(int)
        # Step 2: 
        if self.regressor:
            pred_values = self.regressor.predict(X)
            pred_values = np.maximum(pred_values, 0)
        else:
            pred_values = np.zeros(len(X))
        return is_existing * pred_values

# ==========================================
# 2. ：
# ==========================================
def align_features(df_to_predict, train_features):
    """
     One-Hot 
    """
    # 0
    for col in train_features:
        if col not in df_to_predict.columns:
            df_to_predict[col] = 0
    # ，
    return df_to_predict[train_features]

# ==========================================
# 3. ：
# ==========================================
def generate_final_result():
    print("===  2025  ===")
    
    # 1.  ( feature_2025.csv)
    feature_file = 'output/predicted_feature_2025.csv'
    if not os.path.exists(feature_file):
        print(f"： {feature_file}")
        return
    df = pd.read_csv(feature_file)
    print(f"1. : {len(df)} ")
    
    # 2.  (!)
    # : year, count, SOVI... 
    df['year'] = 2025
    
    # 
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    #  (Interaction Features)
    df['pop_x_count'] = df['pop_density'] * df['count']
    df['sovi_x_count'] = df['SOVI_SCORE'] * df['count']
    
    # 3. One-Hot 
    #  ()
    numerical_features = [
        'year', 'count', 'SOVI_SCORE', 'RESL_SCORE',
        'pop_density', 'avg_eal_valt',
        'pop_x_count', 'sovi_x_count'
    ]
    categorical_features = ['state_abbr', 'disaster_name']
    
    print("2.  One-Hot ...")
    X_pred = pd.get_dummies(
        df[numerical_features + categorical_features],
        columns=categorical_features,
        dtype=float
    )
    # dummy(month_sin)
    # month_sindummy，，
    # ，train_one_target  month_sin/cos  numerical_features
    #  load_and_merge_data ，。， numerical_features 。
    
    # 4. 
    print("3. ...")
    try:
        model_loss = joblib.load("output_model/two_stage_xgb_Loss.pkl")
        model_fatal = joblib.load("output_model/two_stage_xgb_Fatality.pkl")
    except Exception as e:
        print(f": {e}")
        return

    # 5. 
    # 
    try:
        #  booster 
        train_features = model_loss.classifier.get_booster().feature_names
    except:
        print("：， ()...")
        train_features = X_pred.columns.tolist()
    
    X_final = align_features(X_pred, train_features)
    
    # 6.  (Predict)
    print("4.  Avg Loss  Avg Fatality...")
    
    #  Loss ( Log)
    pred_loss_log = model_loss.predict(X_final)
    pred_avg_loss = np.expm1(pred_loss_log)
    pred_avg_loss = np.maximum(pred_avg_loss, 0)
    
    #  Fatality ( Log)
    pred_fatal_log = model_fatal.predict(X_final)
    pred_avg_fatality = np.expm1(pred_fatal_log)
    pred_avg_fatality = np.maximum(pred_avg_fatality, 0)
    
    # 7.  & 
    df['pred_avg_loss'] = pred_avg_loss
    df['pred_avg_fatality'] = pred_avg_fatality
    
    #  Total Loss  Total Fatality
    # Total = Avg * Count
    df['pred_total_loss'] = df['pred_avg_loss'] * df['count']
    df['pred_total_fatality'] = df['pred_avg_fatality'] * df['count']
    
    # 8. 
    output_file = 'output/result_2025.csv'
    
    # ，
    cols_order = [
        'year', 'month', 'state_abbr', 'disaster_name', 'count',
        'pred_avg_loss', 'pred_total_loss',
        'pred_avg_fatality', 'pred_total_fatality',
        'SOVI_SCORE', 'RESL_SCORE', 'pop_density', 'avg_eal_valt'
    ]
    # 
    cols_order = [c for c in cols_order if c in df.columns]
    
    df[cols_order].to_csv(output_file, index=False)
    
    print(f"\n✅ ！: {output_file}")
    print(f"   - : {len(df)}")
    print("\n--- Top 5  ---")
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(df[cols_order].sort_values('pred_total_loss', ascending=False).head(5))

if __name__ == "__main__":
    generate_final_result()