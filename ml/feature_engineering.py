import pandas as pd
from simpledbf import Dbf5
import numpy as np

# -----------------------------
# 1. 
# -----------------------------
print("...")

noaa = pd.read_csv('dataset/US_Disasters_2000_2024.csv')  # 
nri = Dbf5('dataset/NRI_Shapefile_CensusTracts.dbf').to_dataframe()

# -----------------------------
# 2.  NOAA  STATE 
# -----------------------------
#  NOAA  STATE （ "ALABAMA"）
state_name_to_abbr = {
    'ALABAMA': 'AL', 'ALASKA': 'AK', 'ARIZONA': 'AZ', 'ARKANSAS': 'AR',
    'CALIFORNIA': 'CA', 'COLORADO': 'CO', 'CONNECTICUT': 'CT', 'DELAWARE': 'DE',
    'FLORIDA': 'FL', 'GEORGIA': 'GA', 'HAWAII': 'HI', 'IDAHO': 'ID',
    'ILLINOIS': 'IL', 'INDIANA': 'IN', 'IOWA': 'IA', 'KANSAS': 'KS',
    'KENTUCKY': 'KY', 'LOUISIANA': 'LA', 'MAINE': 'ME', 'MARYLAND': 'MD',
    'MASSACHUSETTS': 'MA', 'MICHIGAN': 'MI', 'MINNESOTA': 'MN', 'MISSISSIPPI': 'MS',
    'MISSOURI': 'MO', 'MONTANA': 'MT', 'NEBRASKA': 'NE', 'NEVADA': 'NV',
    'NEW HAMPSHIRE': 'NH', 'NEW JERSEY': 'NJ', 'NEW MEXICO': 'NM', 'NEW YORK': 'NY',
    'NORTH CAROLINA': 'NC', 'NORTH DAKOTA': 'ND', 'OHIO': 'OH', 'OKLAHOMA': 'OK',
    'OREGON': 'OR', 'PENNSYLVANIA': 'PA', 'RHODE ISLAND': 'RI', 'SOUTH CAROLINA': 'SC',
    'SOUTH DAKOTA': 'SD', 'TENNESSEE': 'TN', 'TEXAS': 'TX', 'UTAH': 'UT',
    'VERMONT': 'VT', 'VIRGINIA': 'VA', 'WASHINGTON': 'WA', 'WEST VIRGINIA': 'WV',
    'WISCONSIN': 'WI', 'WYOMING': 'WY'
}

noaa['state_abbr'] = noaa['STATE'].str.upper().map(state_name_to_abbr)
noaa = noaa.dropna(subset=['state_abbr'])  # 

# -----------------------------
# 3.  NRI 
# -----------------------------
# 
state_agg_dict = {
    'POPULATION': 'sum',
    'BUILDVALUE': 'sum',
    'AGRIVALUE': 'sum',
    'AREA': 'sum',
    'SOVI_SCORE': 'mean',
    'RESL_SCORE': 'mean',
    'EAL_VALT': 'sum',    # 
    'EAL_VALB': 'sum',    # 
    'EAL_VALP': 'sum',    # 
}

# （：Thunderstorm Wind → SWND_*）
hazard_prefix_map = {
    'Thunderstorm Wind': 'SWND',
    'Tornado': 'TRND',
    'Heat': 'HWAV',
    'Ice Storm': 'ISTM',
    'Winter Storm': 'WNTW',
    'High Wind': 'SWND',  # 
    'Hail': 'HAIL',
    'Flood': 'CFLD',      # Coastal Flood， RFLD
    'Riverine Flooding': 'RFLD',
    'Hurricane': 'HRCN',
    'Wildfire': 'WFIR',
    'Earthquake': 'ERQK',
    'Drought': 'DRGT',
    'Lightning': 'LTNG',
    'Landslide': 'LNDS',
    'Cold Wave': 'CWAV',
    'Avalanche': 'AVLN',
    'Tsunami': 'TSUN',
    'Volcanic Activity': 'VLCN'
}

# （ SWND_EVNTS, SWND_EALT ）

# 
nri_state = nri.groupby('STATEABBRV').agg(state_agg_dict).reset_index()
nri_state.rename(columns={'STATEABBRV': 'state_abbr'}, inplace=True)

# （）
nri_state['pop_density'] = nri_state['POPULATION'] / (nri_state['AREA'] + 1e-8)
nri_state['eal_per_capita'] = nri_state['EAL_VALT'] / (nri_state['POPULATION'] + 1e-8)
nri_state['eal_per_buildvalue'] = nri_state['EAL_VALB'] / (nri_state['BUILDVALUE'] + 1e-8)

# -----------------------------
# 4. （ state, month, disaster_name  NOAA）
# -----------------------------
noaa_agg = noaa.groupby(['year', 'state_abbr', 'month', 'disaster_name']).agg(
    count=('year', 'size'),
    sum_loss=('loss', 'sum'),
    sum_fatalities=('fatalities', 'sum')
).reset_index()
noaa_agg['avg_loss'] = noaa_agg['sum_loss'] / noaa_agg['count']
noaa_agg['avg_fatality'] = noaa_agg['sum_fatalities'] / noaa_agg['count']
# -----------------------------
# 5.  NRI（）
# -----------------------------
final_data = []
for _, row in noaa_agg.iterrows():
    state = row['state_abbr']
    nri_row = nri_state[nri_state['state_abbr'] == state]
    if nri_row.empty:
        continue
    nri_row = nri_row.iloc[0].to_dict()
    
    # 
    avg_eal_valt = nri_row['EAL_VALT'] / (nri_row['POPULATION'] + 1e-8)

    feat = {
        'year': row['year'],
        'state_abbr': state,
        'month': row['month'],
        'disaster_name': row['disaster_name'],
        'count': row['count'],
        'SOVI_SCORE': nri_row['SOVI_SCORE'],
        'RESL_SCORE': nri_row['RESL_SCORE'],
        'pop_density': nri_row['pop_density'],
        'avg_eal_valt': avg_eal_valt,
    }

    # 
    feat.update({
        'avg_loss': row['avg_loss'],
        'avg_fatality': row['avg_fatality']
    })
    
    final_data.append(feat)

feature_df = pd.DataFrame(final_data)

# -----------------------------
# 6. 
# -----------------------------
target_cols = [ 'avg_loss', 'avg_fatality']

years = sorted(feature_df['year'].unique())

for y in years:
    df_y = feature_df[feature_df['year'] == y]

    features_y = df_y.drop(columns=target_cols)
    targets_y = df_y[['state_abbr', 'month', 'disaster_name'] + target_cols]

    features_y.to_csv(f'processed_data/feature_{y}.csv', index=False)
    targets_y.to_csv(f'processed_data/target_{y}.csv', index=False)

print("✅  feature_YYYY.csv  target_YYYY.csv")