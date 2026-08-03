"""Download the eight TDC ADMET binary classification datasets.

Usage: python scripts/download_data.py  (writes data/<Endpoint>.csv)
Requires: pip install PyTDC  (network access to the TDC server)
"""
import os
import pandas as pd
from tdc.single_pred import ADME, Tox

ENDPOINTS = {
    'hERG': (Tox, 'hERG'),
    'AMES': (Tox, 'AMES'),
    'BBB_Martins': (ADME, 'BBB_Martins'),
    'Pgp_Broccatelli': (ADME, 'Pgp_Broccatelli'),
    'CYP2C9_Veith': (ADME, 'CYP2C9_Veith'),
    'CYP2D6_Veith': (ADME, 'CYP2D6_Veith'),
    'CYP3A4_Veith': (ADME, 'CYP3A4_Veith'),
    'Bioavailability_Ma': (ADME, 'Bioavailability_Ma'),
}

os.makedirs('data', exist_ok=True)
for name, (group, tdc_name) in ENDPOINTS.items():
    path = f'data/{name}.csv'
    if os.path.exists(path):
        print(f'skip {name} (exists)')
        continue
    df = group(name=tdc_name, path=f'data/tdc_cache_{name}')
    df.get_data().to_csv(path, index=False)
    print(f'{name}: {len(df.get_data())} rows -> {path}')
print('done')
