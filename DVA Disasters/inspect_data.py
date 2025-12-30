import pandas as pd
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATA")
csv_path = os.path.join(data_dir, "US_Disasters_2000_2024.csv")

print("--- CSV INSPECTION ---")
try:
    df = pd.read_csv(csv_path)
    print("Columns:")
    for col in df.columns:
        print(f"- {col}")
    print("\nFirst 2 rows:")
    print(df.head(2).to_string())
except Exception as e:
    print(f"Error reading CSV: {e}")

print("\n--- SHAPEFILE INSPECTION ---")
try:
    import geopandas as gpd
    print("Geopandas is installed.")
    shp_path = os.path.join(data_dir, "NRI_Shapefile_CensusTracts/NRI_Shapefile_CensusTracts.shp")
    gdf = gpd.read_file(shp_path)
    print("Columns:")
    for col in gdf.columns[:10]: # Print first 10 columns to avoid spam
        print(f"- {col}")
    print("... and more")
    print("\nFirst 2 rows:")
    print(gdf.head(2).to_string())
except ImportError:
    print("Geopandas NOT installed.")
except Exception as e:
    print(f"Error reading Shapefile: {e}")
