import json
import os
from dbfread import DBF
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
DBF_PATH = os.path.join(BASE_DIR, "data/NRI_Shapefile_CensusTracts/NRI_Shapefile_CensusTracts.dbf")
OUTPUT_JSON = os.path.join(DATASET_DIR, "nri_data.json")

# Hazards to process
HAZARDS = {
    'Riverine Flooding': 'RFLD',
    'Hurricane': 'HRCN',
    'Wildfire': 'WFIR',
    'Earthquake': 'ERQK',
    'Tornado': 'TRND',
    'Coastal Flooding': 'CFLD',
    'Strong Wind': 'SWND',
    'Hail': 'HAIL'
}

def process_nri():
    state_data = defaultdict(lambda: {
        'count': 0,
        'sovi_sum': 0.0,
        'resl_sum': 0.0,
        'risk_sum': 0.0,
        'eal_total': 0.0,
        'hazards': defaultdict(float)
    })
    
    try:
        table = DBF(DBF_PATH, load=False, encoding='cp1252')
        fields = set(table.field_names)
        
        for i, record in enumerate(table):
            if i % 10000 == 0:
                print(f"Processed {i} records...")
                
            state = record['STATE']
            if not state: continue
            state = state.title()
            
            sovi = record.get('SOVI_SCORE')
            resl = record.get('RESL_SCORE')
            risk = record.get('RISK_SCORE')
            eal_t = record.get('EAL_VALT')
            
            s_data = state_data[state]
            s_data['count'] += 1
            
            if sovi is not None: s_data['sovi_sum'] += float(sovi)
            if resl is not None: s_data['resl_sum'] += float(resl)
            if risk is not None: s_data['risk_sum'] += float(risk)
            if eal_t is not None: s_data['eal_total'] += float(eal_t)
            
            for h_name, prefix in HAZARDS.items():
                val = 0.0
                if f"{prefix}_EALB" in fields:
                    v = record.get(f"{prefix}_EALB")
                    if v: val += float(v)
                    
                if f"{prefix}_EALA" in fields:
                    v = record.get(f"{prefix}_EALA")
                    if v: val += float(v)
                    
                s_data['hazards'][h_name] += val
                
    except Exception as e:
        print(f"Error processing DBF: {e}")
        return

    print("Aggregating final results...")
    final_output = {}
    
    for state, data in state_data.items():
        count = data['count']
        if count == 0: continue
        
        final_output[state] = {
            "risk_score": data['risk_sum'] / count,
            "sovi_score": data['sovi_sum'] / count,
            "resl_score": data['resl_sum'] / count,
            "eal_total": data['eal_total'],
            "hazards": dict(data['hazards'])
        }
        
    print(f"Writing to {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(final_output, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    process_nri()
