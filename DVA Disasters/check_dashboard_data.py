import json

import os

# Load dashboard data
json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard-app", "dashboard_data.json")
with open(json_path, 'r') as f:
    data = json.load(f)

print("=== DASHBOARD DATA STRUCTURE ===")
print(f"Data type: {type(data)}")

if isinstance(data, dict):
    print(f"Top-level keys: {list(data.keys())}")
elif isinstance(data, list):
    print(f"Data is a list with {len(data)} entries")
    if data:
        print(f"First entry keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
print()

# Check historical data
if 'historical' in data:
    hist = data['historical']
    states = list(hist.keys())
    print(f"Number of states in historical: {len(states)}")
    print(f"First 5 states: {states[:5]}")
    
    if states:
        first_state = states[0]
        print(f"\nStructure for '{first_state}':")
        state_data = hist[first_state]
        
        if isinstance(state_data, dict):
            years = list(state_data.keys())
            print(f"  Years available: {years[:5]}...")
            
            if years:
                first_year = years[0]
                year_data = state_data[first_year]
                print(f"\n  Data for year {first_year}:")
                print(f"    Keys: {list(year_data.keys())}")
                
                if 'events' in year_data:
                    events = year_data['events']
                    print(f"    Event types: {list(events.keys())[:5]}")
                    if events:
                        first_event = list(events.keys())[0]
                        print(f"    Sample event '{first_event}': {events[first_event]}")

# Check prediction data
if 'prediction_2025' in data:
    pred = data['prediction_2025']
    print(f"\nPrediction data available: {len(pred)} states")
    if pred:
        first_state = list(pred.keys())[0]
        print(f"Sample prediction for '{first_state}': {pred[first_state]}")

# Load NRI data
print("\n=== NRI DATA STRUCTURE ===")
nri_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard-app", "nri_data.json")
with open(nri_path, 'r') as f:
    nri_data = json.load(f)

print(f"Type: {type(nri_data)}")
if isinstance(nri_data, list):
    print(f"Number of entries: {len(nri_data)}")
    if nri_data:
        print(f"\nFirst entry keys: {list(nri_data[0].keys())}")
        print(f"Sample entry: {nri_data[0]}")
elif isinstance(nri_data, dict):
    print(f"Keys: {list(nri_data.keys())}")
