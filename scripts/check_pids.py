import pandas as pd
import os

data_dir = r"C:\Users\Asus\.gemini\antigravity-ide\scratch\kestrel-warranty\data"
partners = pd.read_csv(os.path.join(data_dir, "partners.csv"))
p_list = ['SP3160', 'SP3232', 'SP3318', 'SP3129', 'SP3319', 'SP3118', 'SP3286']
print(partners[partners['partner_id'].isin(p_list)])
