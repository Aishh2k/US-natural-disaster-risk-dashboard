import csv
import json
from collections import defaultdict

# Read the CSV file
input_file = 'US_Disasters_Prediction_2025.csv'
output_file = 'datasets/predictions_data.json'

# Data structure: {"2025": {"State": {"loss": X, "fatalities": Y, "events": [...]}}}
predictions_data = {}
predictions_data["2025"] = {}

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        state = row['STATE'].title()  # Convert to Title Case
        
        # Initialize state if not exists
        if state not in predictions_data["2025"]:
            predictions_data["2025"][state] = {
                "loss": 0,
                "fatalities": 0,
                "events": []
            }
        
        # Parse values
        try:
            loss = float(row['predicted_loss']) if row['predicted_loss'] else 0
            fatalities = int(float(row['predicted_fatalities'])) if row['predicted_fatalities'] else 0
            event_type = row['most_likely_disaster']
            month = int(row['month'])
            
            # Add to state totals
            predictions_data["2025"][state]["loss"] += loss
            predictions_data["2025"][state]["fatalities"] += fatalities
            
            # Add event (aggregating by event type AND month per state)
            # Find if this event type+month already exists
            event_found = False
            for event in predictions_data["2025"][state]["events"]:
                if event["type"] == event_type and event.get("month") == month:
                    event["loss"] += loss
                    event["fatalities"] += fatalities
                    event["count"] = event.get("count", 0) + 1
                    event_found = True
                    break
            
            if not event_found:
                predictions_data["2025"][state]["events"].append({
                    "type": event_type,
                    "month": month,
                    "loss": loss,
                    "fatalities": fatalities,
                    "count": 1
                })
        
        except (ValueError, KeyError) as e:
            print(f"Error processing row for {state}: {e}")
            continue

# Sort events by loss (descending) for each state
for state in predictions_data["2025"]:
    predictions_data["2025"][state]["events"].sort(key=lambda x: x["loss"], reverse=True)

# Write to JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(predictions_data, f, indent=2)

print(f"✓ Converted prediction data to {output_file}")
print(f"✓ Total states: {len(predictions_data['2025'])}")

# Calculate total loss and fatalities
total_loss = sum(state["loss"] for state in predictions_data["2025"].values())
total_fatalities = sum(state["fatalities"] for state in predictions_data["2025"].values())
print(f"✓ Total predicted loss for 2025: ${total_loss:,.0f}")
print(f"✓ Total predicted fatalities for 2025: {total_fatalities}")
