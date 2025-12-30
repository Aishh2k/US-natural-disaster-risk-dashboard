import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os
import math
class WeightedMSELoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight  # （）

    def forward(self, pred, target):
        # 1.  MSE (，)
        loss = (pred - target) ** 2
        
        # 2. 
        #  target > 0 ()， pos_weight
        #  target == 0 ()， 1.0
        # ： log1p， target > 0  >= 1
        weights = torch.ones_like(target)
        weights[target > 0] = self.pos_weight
        
        # 3. 
        weighted_loss = loss * weights
        return weighted_loss.mean()
# ==========================================
# 0. ：
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

# ==========================================
# 1. 
# ==========================================
def get_all_disaster_types(data_folder='./processed_data/', years=range(2000, 2025)):
    all_types = set()
    for year in years:
        file_path = f"{data_folder}feature_{year}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'disaster_name' in df.columns:
                all_types.update(df['disaster_name'].dropna().unique())
    return sorted(list(all_types))

# ==========================================
# 2. ：Count  + Log
# ==========================================
def load_and_process_data_multilabel(disaster_types):
    print("...")
    all_data = []
    years = range(2000, 2025)
    
    for year in years:
        file_path = f"processed_data/feature_{year}.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['year'] = year
            all_data.append(df)
            
    if not all_data:
        raise ValueError("！")

    raw_df = pd.concat(all_data, ignore_index=True)
    
    # 1. 
    static_cols = ['SOVI_SCORE', 'RESL_SCORE', 'pop_density', 'avg_eal_valt']
    for col in static_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors='coerce').fillna(0)

    # 2.  count 
    if 'count' not in raw_df.columns:
        print("： count ， 1")
        raw_df['count'] = 1
    else:
        raw_df['count'] = pd.to_numeric(raw_df['count'], errors='coerce').fillna(0)

    # 3. ： (disaster_name, count) 
    #  "Avalanche: 1"  "Avalanche"  1
    pivot_df = raw_df.pivot_table(
        index=['year', 'month', 'state_abbr'],
        columns='disaster_name',
        values='count',
        aggfunc='sum',
        fill_value=0
    ).reset_index()
    
    # 4.  (50 x 25 x 12)
    states = raw_df['state_abbr'].unique()
    months = range(1, 13)
    grid = []
    
    # 
    state_static_map = raw_df.groupby('state_abbr')[static_cols].first().reset_index()

    for y in years:
        for m in months:
            for s in states:
                grid.append({'year': y, 'month': m, 'state_abbr': s})
    
    grid_df = pd.DataFrame(grid)
    grid_df = grid_df.merge(state_static_map, on='state_abbr', how='left')
    
    # 5. 
    full_df = grid_df.merge(pivot_df, on=['year', 'month', 'state_abbr'], how='left')
    
    # 6.  ()
    for dtype in disaster_types:
        if dtype not in full_df.columns:
            full_df[dtype] = 0.0
        full_df[dtype] = full_df[dtype].fillna(0).astype(float)
        
    full_df[static_cols] = full_df[static_cols].fillna(0).astype(float)
    
    # === ：Log  () ===
    #  log1p (ln(x+1))
    #  800  6.7，1  0.69，
    full_df[disaster_types] = np.log1p(full_df[disaster_types])
    
    # 7. 
    scaler = MinMaxScaler()
    full_df[static_cols] = scaler.fit_transform(full_df[static_cols])
    
    # 8. 
    full_df['month_sin'] = np.sin(2 * np.pi * full_df['month'] / 12)
    full_df['month_cos'] = np.cos(2 * np.pi * full_df['month'] / 12)
    
    # 
    full_df = full_df.sort_values(['state_abbr', 'year', 'month']).reset_index(drop=True)
    
    print("。")
    return full_df, scaler, states

# ==========================================
# 3.  ()
# ==========================================
class MultiLabelDisasterTransformer(nn.Module):
    def __init__(self, num_features, num_classes, d_model=64, nhead=4, num_layers=2, output_len=12):
        super().__init__()
        self.input_linear = nn.Linear(num_features, d_model)
        self.input_dropout = nn.Dropout(p=0.1)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.flatten = nn.Flatten()
        self.pre_output_dropout = nn.Dropout(p=0.1)
        
        self.output_head = nn.Linear(d_model * 60, output_len * num_classes) 
        
        self.num_classes = num_classes
        self.output_len = output_len

    def forward(self, x):
        x = self.input_linear(x)
        x = self.input_dropout(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.flatten(x)
        x = self.pre_output_dropout(x)
        out = self.output_head(x) 
        out = out.view(-1, self.output_len, self.num_classes)
        return out #  Log 

# ==========================================
# 4. Dataset
# ==========================================
class DisasterTimeSeriesDataset(Dataset):
    def __init__(self, data, input_len=60, pred_len=12, feature_cols=None, disaster_cols=None):
        self.sequences = []
        self.targets = []
        self.target_years = []
        
        states = data['state_abbr'].unique()
        
        for state in states:
            state_data = data[data['state_abbr'] == state].reset_index(drop=True)
            X_vals = state_data[feature_cols].values
            y_vals = state_data[disaster_cols].values 
            years = state_data['year'].values
            
            total_len = len(state_data)
            for i in range(total_len - input_len - pred_len + 1):
                seq_x = X_vals[i : i + input_len]
                seq_y = y_vals[i + input_len : i + input_len + pred_len]
                target_start_year = years[i + input_len]
                
                self.sequences.append(seq_x)
                self.targets.append(seq_y)
                self.target_years.append(target_start_year)
                
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return (torch.tensor(self.sequences[idx], dtype=torch.float32), 
                torch.tensor(self.targets[idx], dtype=torch.float32))

# ==========================================
# 5.  ( Log )
# ==========================================
def generate_predictions(model, full_df, all_states, feature_cols, disaster_types, target_year, device, save_filename):
    input_start_year = target_year - 5
    input_end_year = target_year - 1
    print(f"\n {target_year} ...")
    predictions = []
    has_ground_truth = target_year <= full_df['year'].max()
    
    with torch.no_grad():
        for state in all_states:
            state_input_df = full_df[
                (full_df['state_abbr'] == state) & 
                (full_df['year'] >= input_start_year) & 
                (full_df['year'] <= input_end_year)
            ].sort_values(['year', 'month'])
            
            if len(state_input_df) < 60: continue
            
            input_data = state_input_df[feature_cols].values
            input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).to(device)
            
            logits = model(input_tensor)
            
            # ===  Log  ===
            #  Log， Log， expm1 
            pred_counts = torch.expm1(logits).clamp(min=0).squeeze(0).cpu().numpy()
            
            ground_truth = None
            if has_ground_truth:
                state_target_df = full_df[
                    (full_df['state_abbr'] == state) & 
                    (full_df['year'] == target_year)
                ].sort_values('month')
                if len(state_target_df) == 12:
                    # ：full_df  Log ，，
                    ground_truth = np.expm1(state_target_df[disaster_types].values)
            
            for month_idx in range(12):
                for type_idx, disaster_name in enumerate(disaster_types):
                    val = float(pred_counts[month_idx, type_idx])
                    
                    record = {
                        'state': state,
                        'year': target_year,
                        'month': month_idx + 1,
                        'disaster_type': disaster_name,
                        'predicted_count': val
                    }
                    if ground_truth is not None:
                        record['actual_count'] = float(ground_truth[month_idx, type_idx])
                    predictions.append(record)
    
    if predictions:
        pd.DataFrame(predictions).to_csv(save_filename, index=False)
        print(f"✅  {save_filename}")

# ==========================================
# 6. 
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    disaster_types = get_all_disaster_types() 
    df, scaler, all_states = load_and_process_data_multilabel(disaster_types)
    
    # === ： disaster_types ===
    #  disaster_types， count  count
    basic_features = ['SOVI_SCORE', 'RESL_SCORE', 'pop_density', 'avg_eal_valt', 'month_sin', 'month_cos']
    feature_cols = basic_features + disaster_types
    
    num_features = len(feature_cols)
    num_classes = len(disaster_types)
    print(f": {num_features} ( {num_classes} )")

    full_dataset = DisasterTimeSeriesDataset(df, input_len=60, pred_len=12, 
                                           feature_cols=feature_cols, 
                                           disaster_cols=disaster_types)
    
    # 
    train_indices = []
    val_indices = []
    for idx, year in enumerate(full_dataset.target_years):
        if year == 2024:
            val_indices.append(idx)
        elif year < 2024:
            train_indices.append(idx)
            
    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)
    train_loader = DataLoader(full_dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(full_dataset, batch_size=32, sampler=val_sampler)
    
    print(f": {len(train_indices)} | : {len(val_indices)}")
    
    model = MultiLabelDisasterTransformer(num_features, num_classes, d_model=64, num_layers=4).to(device)
    
    #  MSE Loss ( Log ，)
    criterion = WeightedMSELoss(pos_weight=5.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_val_loss = float('inf')
    patience = 15
    counter = 0
    train_loss_history = [] 
    val_loss_history = [] 
    batch_loss_history=[]
    batch_loss_val_history=[]
    for epoch in range(50):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y) #  y  log 
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            batch_loss_history.append(loss.item())
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = criterion(output, y)
                val_loss += loss.item()
                batch_loss_val_history.append(loss.item())
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_loss_history.append(avg_train) 
        val_loss_history.append(avg_val)
        print(f"Epoch {epoch+1} | Train MSE(Log): {avg_train:.4f} | Val MSE(Log): {avg_val:.4f}")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'output_model/Transformer_best_model.pth')
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print("")
                break
    import matplotlib.pyplot as plt 
    plt.figure(figsize=(12,5)) 
    plt.plot(batch_loss_history, label='Train Loss') 
    plt.plot(batch_loss_val_history, label='Validation Loss') 
    plt.xlabel("Batch") 
    plt.ylabel("Loss") 
    plt.title("Training Loss Per Batch") 
    plt.grid(True) 
    plt.savefig("output/transformer_loss_curve_batch.png", dpi=300) 
    plt.close() 
    print("Batch  loss  loss_curve_batch.png") 
    plt.figure(figsize=(8, 5)) 
    plt.plot(train_loss_history, label='Train Loss') 
    plt.plot(val_loss_history, label='Validation Loss') 
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.title('Training & Validation Loss Curve') 
    plt.legend() 
    plt.grid(True) 
    plt.savefig('output/transformer_loss_curve.png', dpi=300) 
    plt.close()
    print("Loss  loss_curve.png")
    print("\n...")
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    #  2025
    generate_predictions(model, df, all_states, feature_cols, disaster_types, 
                         target_year=2025, device=device, save_filename='output/prediction_2025_counts.csv')
    
    # 
    print("\n...")
    preds_df = pd.read_csv('output/prediction_2025_counts.csv')
    #  > 0.5  ( Log  0.5 )
    high_risk = preds_df[preds_df['predicted_count'] > 0.5].sort_values('predicted_count', ascending=False)
    high_risk.to_csv('output/prediction_2025_final_high_risk.csv', index=False)
    print("Top 10 :")
    print(high_risk.head(10))

if __name__ == "__main__":
    main()