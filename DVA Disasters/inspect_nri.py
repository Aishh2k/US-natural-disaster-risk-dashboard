from dbfread import DBF
import os

dbf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATA", "NRI_Shapefile_CensusTracts", "NRI_Shapefile_CensusTracts.dbf")

try:
    # Read only the header
    table = DBF(dbf_path, load=False)
    print("Columns:", table.field_names)
    
    # Identify Hazard EAL columns
    eal_cols = [c for c in table.field_names if '_EALb' in c or '_EALt' in c or '_EALp' in c or '_EALv' in c or '_EALa' in c or '_EALs' in c or '_EALr' in c or '_EAL' in c]
    # Filter for the main composite or specific hazard EALs (usually just _EAL for total or _EALb/p for breakdown)
    # Let's just look for the main ones first.
    
    print("\n--- POTENTIAL HAZARD PREFIXES ---")
    # Common NRI prefixes: RFLD, HWAV, HRCN, WFIR, ERQK, TRND, SWND, HAIL, LNDS, VLCN, DRGT, ISTM, CFLD, TSU, WNTW
    prefixes = ['RFLD', 'HWAV', 'HRCN', 'WFIR', 'ERQK', 'TRND', 'SWND', 'HAIL', 'LNDS', 'VLCN', 'DRGT', 'ISTM', 'CFLD', 'TSU', 'WNTW']
    
    for p in prefixes:
        cols = [c for c in table.field_names if c.startswith(p) and 'EAL' in c]
        if cols:
            print(f"{p}: {cols}")

    print("\n--- RISK / SOVI / RESL ---")
    print("SoVI:", [c for c in table.field_names if 'SOVI' in c])
    print("RESL:", [c for c in table.field_names if 'RESL' in c])
    print("RISK:", [c for c in table.field_names if 'RISK' in c])
    
except Exception as e:
    print(f"Error: {e}")
