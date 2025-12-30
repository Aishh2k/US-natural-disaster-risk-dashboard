import pandas as pd
import numpy as np
import math
import os

def create_feature_2025_file():
    print("===  feature_2025.csv ===")

    # 1.  2025 
    # ：， 'demo_pred.xlsx - Sheet1.csv'
    pred_file = 'output/prediction_2025_counts.csv' 
    if not os.path.exists(pred_file):
        print(f"： {pred_file}")
        return
    df_pred = pd.read_csv(pred_file)
    print(f"1. : {len(df_pred)} ")

    # 2.  +  (Data Cleaning)
    THRESHOLD = 0.5 #  (0.51)
    print(f"   -  (Threshold = {THRESHOLD}) ...")
    
    # ===  1:  floor  ===
    # ： apply  math.floor
    df_pred['predicted_count'] = df_pred['predicted_count'].apply(
        lambda x: math.floor(x) if x > THRESHOLD else 0
    )

    # 3. 
    rename_map = {
        'state': 'state_abbr',           
        'disaster_type': 'disaster_name',
        'predicted_count': 'count'       
    }
    df_pred.rename(columns=rename_map, inplace=True)

    # ===  2: ！ count  0  ===
    # ：
    df_pred = df_pred[df_pred['count'] > 0].copy()
    print(f"   -  0 : {len(df_pred)}  ()")

    # 4.  2000  ()
    feat_file = 'processed_data/feature_2000.csv' # 
    if not os.path.exists(feat_file):
        print(f"： {feat_file}")
        return
    df_feat_source = pd.read_csv(feat_file)
    
    # 5. 
    static_cols = ['state_abbr', 'SOVI_SCORE', 'RESL_SCORE', 'pop_density', 'avg_eal_valt']
    state_static_map = df_feat_source[static_cols].drop_duplicates('state_abbr')
    print(f"2. :  {len(state_static_map)} ")

    # 6.  (Merge)
    #  df_pred  count > 0 ，
    df_2025 = pd.merge(df_pred, state_static_map, on='state_abbr', how='left')

    # 7. 
    df_2025 = df_2025.dropna(subset=['SOVI_SCORE'])
    
    # 8. 
    target_columns = ['year', 'state_abbr', 'month', 'disaster_name', 'count', 
                      'SOVI_SCORE', 'RESL_SCORE', 'pop_density', 'avg_eal_valt']
    
    df_2025['year'] = 2025
    df_final = df_2025[target_columns]
    
    # 9. 
    output_file = 'output/predicted_feature_2025.csv'
    df_final.to_csv(output_file, index=False)
    print(f"\n✅ : {output_file}")
    print(f"   - : {len(df_final)}")
    print("   - 5:")
    print(df_final.head())

if __name__ == "__main__":
    create_feature_2025_file()