# run_prediction_example.py

import pandas as pd
import json
from predict import load_models, predict_full_package
import argparse
import utils
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--reports_dir", type=str, default=None)

    return parser.parse_args()
def run_example():
    args = parse_args()
    utils.init_paths(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        reports_dir=args.reports_dir,
    )
    print("--- Running Prediction Example ---")

    models = load_models()
    
    if models is None:
        print("Failed to load models. Exiting.")
        return
    sample_data = {
        'disaster_type_flood': [1],
        'disaster_type_wind': [0],
        'magnitude_normalized': [0.85],
        'duration_hours': [12.5],
        'season_month': [6],
        'season_quarter': [2],
        'is_urban': [1],
        'population_density_log': [5.1],
        'gdp_per_capita_normalized': [0.75],
        'historical_disaster_frequency': [5],
        'preparedness_index': [0.6],
        'early_warning_system_level': [1],
        'healthcare_access_index': [0.8],
        'past_response_time_hours': [4.5]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    print("\nSample Input Features:")
    print(sample_df)
    
    prediction_results = predict_full_package(models, sample_df)
    
    print("\n--- Full Prediction Package (Output) ---")
    print(json.dumps(prediction_results, indent=4))
if __name__ == "__main__":
    run_example()