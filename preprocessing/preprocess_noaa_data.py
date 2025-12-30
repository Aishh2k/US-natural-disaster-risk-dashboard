import json
import os
import pandas as pd

from ml.predict import predict_next_year

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "US_Disasters_2000_2024.csv")
OUTPUT_JSON = os.path.join(DATASET_DIR, "noaa_data.json")

def load_and_clean_data():
    df = pd.read_csv(CSV_PATH)

    df.columns = [c.lower().strip() for c in df.columns]

    if df['loss'].dtype == 'O':
        df['loss'] = df['loss'].astype(str).str.replace(',', '').astype(float)
    
    if df['fatalities'].dtype == 'O':
        df['fatalities'] = df['fatalities'].astype(str).str.replace(',', '').astype(float)

    df['loss'] = df['loss'].fillna(0)
    df['fatalities'] = df['fatalities'].fillna(0)

    df['year'] = df['year'].astype(int)

    df['state'] = df['state'].astype(str).str.title()

    if 'month' in df.columns:
        df['month'] = df['month'].fillna(0).astype(int)
    
    return df

def aggregate_data(df):
    pre_agg = df.groupby(['year', 'state', 'disaster_name', 'month'])[['loss', 'fatalities']].sum().reset_index()
    
    grouped = pre_agg.groupby(['year', 'state'])
    
    agg_data = {}
    
    for (year, state), group in grouped:
        total_loss = group['loss'].sum()
        total_fatalities = group['fatalities'].sum()
        
        top_events_series = group.groupby('disaster_name')['loss'].sum().sort_values(ascending=False).head(3)
        top_events = top_events_series.index.tolist()

        events_list = []
        for _, row in group.iterrows():
            events_list.append({
                'type': row['disaster_name'],
                'name': row['disaster_name'],
                'loss': float(row['loss']),
                'fatalities': int(row['fatalities']),
                'month': int(row['month'])
            })
        
        year_str = str(year)
        if year_str not in agg_data:
            agg_data[year_str] = {}
        
        agg_data[year_str][state] = {
            "loss": float(total_loss),
            "fatalities": float(total_fatalities),
            "top_events": top_events,
            "events": events_list
        }
        
    return agg_data

def get_unique_event_types(df):
    if 'disaster_name' in df.columns:
        return sorted(df['disaster_name'].unique().tolist())
    return []

def main():
    try:
        df = load_and_clean_data()
        historical_data = aggregate_data(df)
        event_types = get_unique_event_types(df)
        predictions = predict_next_year(df)

        final_output = {
            "historical": historical_data,
            "predictions": predictions,
            "unique_event_types": event_types
        }
        
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(final_output, f, indent=2)
            
        print(f"Successfully created {OUTPUT_JSON}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
