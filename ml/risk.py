import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

# ==========================================
# 1. : 
# ==========================================
def create_risk_tiers(y_raw, high_percentile=0.90):
    """
     (Low, Medium, High)。
    :
    - Class 0 (Low):  0  (/)
    - Class 1 (Medium): ， high_percentile  ()
    - Class 2 (High): ， high_percentile  ()
    """
    labels = np.zeros_like(y_raw, dtype=int)
    
    # 0
    positive_mask = y_raw > 1e-6
    positive_values = y_raw[positive_mask]
    
    if len(positive_values) == 0:
        print(": ， Class 0")
        return labels
        
    #  ( Top 10%)
    threshold_high = np.quantile(positive_values, high_percentile)
    
    print(f"   [Tiering Strategy]")
    print(f"   - Class 0 (Low)   : Value = 0")
    print(f"   - Class 1 (Medium): 0 < Value <= {threshold_high:,.2f}")
    print(f"   - Class 2 (High)  : Value > {threshold_high:,.2f} (Top {100*(1-high_percentile):.0f}% of disasters)")

    # 
    #  1
    labels[positive_mask] = 1
    #  2
    labels[y_raw > threshold_high] = 2
    
    return labels

# ==========================================
# 2.  ()
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
            
            # 
            df_feat.columns = [c.strip() for c in df_feat.columns]
            df_targ.columns = [c.strip() for c in df_targ.columns]
            
            merge_keys = ['state_abbr', 'month', 'disaster_name']
            merged_df = pd.merge(df_feat, df_targ, on=merge_keys, how='inner')
            
            if 'year' not in merged_df.columns:
                merged_df['year'] = year
            
            all_merged_data.append(merged_df)
    
    if not all_merged_data:
        print(": ， processed_data 。")
        return None 
        
    full_df = pd.concat(all_merged_data, ignore_index=True)
    return full_df

# ==========================================
# 3. 
# ==========================================
def train_classification_task(target_col_name, display_name="Loss"):
    print(f"\n========================================")
    print(f"🚀 : {display_name}")
    print(f"========================================")
    
    # 1. 
    data = load_and_merge_data()
    if data is None: return

    # 2.  ()
    data['pop_x_count'] = data['pop_density'] * data['count']
    data['sovi_x_count'] = data['SOVI_SCORE'] * data['count']
    
    numerical_features = [
        'year', 'count', 'SOVI_SCORE', 'RESL_SCORE',
        'pop_density', 'avg_eal_valt',
        'pop_x_count', 'sovi_x_count'
    ]
    categorical_features = ['state_abbr', 'disaster_name']
    
    # 
    data[numerical_features] = data[numerical_features].fillna(0)
    
    # One-Hot Encoding
    X = pd.get_dummies(
        data[numerical_features + categorical_features],
        columns=categorical_features,
        dtype=float
    )
    
    # 3.  Target ()
    y_raw = data[target_col_name].fillna(0).values
    y_class = create_risk_tiers(y_raw, high_percentile=0.90) # Top 10%  High
    
    # 
    unique, counts = np.unique(y_class, return_counts=True)
    dist = dict(zip(unique, counts))
    print(f"   : {dist} (0=Low, 1=Med, 2=High)")

    # 4. 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    # : stratify=y_class  High 

    # 5.  XGBoost Classifier
    # objective='multi:softprob' 
    clf = xgb.XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        objective='multi:softprob',
        num_class=3,  # 3
        n_jobs=-1,
        random_state=42,
        # class_weight ， sample_weight ， scale_pos_weight ()
        # ， class_weight，， sample_weight
    )
    
    print("    XGBoost Classifier...")
    clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    # 6. 
    y_pred = clf.predict(X_test)
    
    print("\n📊 Classification Report ():")
    target_names = ['Low (Class 0)', 'Medium (Class 1)', 'High (Class 2)']
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 7.  (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title(f'Confusion Matrix - {display_name} Prediction')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    
    save_path_cm = f'output/confusion_matrix_{display_name}.png'
    plt.savefig(save_path_cm)
    plt.close()
    print(f"✅ : {save_path_cm}")

    # 8. 
    model_path = f'output_model/classifier_{display_name}.pkl'
    joblib.dump(clf, model_path)
    print(f"✅ : {model_path}")

# ==========================================
# 4. 
# ==========================================
if __name__ == "__main__":
    # 
    os.makedirs('output', exist_ok=True)
    os.makedirs('output_model', exist_ok=True)
    
    #  1: 
    # 'avg_loss'  target CSV ， CSV 
    train_classification_task(target_col_name='avg_loss', display_name="Economic_Loss")
    
    #  2: 
    # 'avg_fatality'  target CSV 
    train_classification_task(target_col_name='avg_fatality', display_name="Fatality")