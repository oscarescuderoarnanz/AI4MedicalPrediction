"""

Description:

   It loads all the files included in the FILENAMES variable and
   selects only the original features that have not been created
   through feature engineering.

   The excluding suffixes are:
    _ind, _locf, _cnt, 4lbk, 8lbk and 16lbk

"""
# Libraries
import pandas as pd


# -----------------------------------------------
# Example with real data
# -----------------------------------------------
# Libraries
from pathlib import Path

EXCLUDE_SUFFIXES = [
    '_ind', '_locf', 'cnt', '4lbk', '8lbk', '16lbk'
]

# Data
DATAPATH = Path('C:\\Users\\kelda\\Desktop\\datasets\\ricu\\mac')

# Filenames
FILENAMES = [
    #'eicu_demo_0.5.6.parquet',
    #'eicu_0.5.6.parquet',
    'mimic_demo_0.5.6.parquet',
    'mimic_0.5.6.parquet',
    'hirid_0.5.6.parquet',
    'miiv_0.5.6.parquet',
    'aumc_0.5.6.parquet',
]

for f in FILENAMES:

    # Define full path
    path = DATAPATH / f
    # Show
    print("Loading... <%s>" % f)
    # Load data
    df = pd.read_parquet(path)

    # Select columns
    cols = []
    for c in df.columns:
        isin = False
        for suffix in EXCLUDE_SUFFIXES:
            if suffix in c:
                isin = True
                continue

        if not isin:
            cols.append(c)

    # Select columns
    aux = df[cols]

    # Format
    aux.columns = aux.columns.str.removesuffix('_raw').values

    # Select and save
    #aux.to_csv(DATAPATH / ('%s_raw.csv' % path.stem))
    aux.to_parquet(DATAPATH / ('%s_raw.parquet' % path.stem))