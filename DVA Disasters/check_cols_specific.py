from dbfread import DBF
dbf_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "DATA", "NRI_Shapefile_CensusTracts", "NRI_Shapefile_CensusTracts.dbf")
table = DBF(dbf_path, load=False)
rfld_cols = [c for c in table.field_names if c.startswith('RFLD_') and 'EAL' in c]
print("RFLD EAL Columns:", rfld_cols)

# Check for other hazards
hrcn_cols = [c for c in table.field_names if c.startswith('HRCN_') and 'EAL' in c]
print("HRCN EAL Columns:", hrcn_cols)

# Check for total EAL
total_eal = [c for c in table.field_names if c.startswith('EAL_')]
print("Total EAL Columns:", total_eal)
