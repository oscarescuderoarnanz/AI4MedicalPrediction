# Libraries
import pandas as pd
import numpy as np

# -----------------------------------------------
# Example with real data
# -----------------------------------------------
# Libraries
from pathlib import Path

# Data
DATAPATH = Path('C:\\Users\\kelda\\Desktop\\datasets\\ricu\\mac')

# Columns to visualise from loaded data.
cols = ['stay_id', 'stay_time', 'abx', 'sepshk_der']

# .. note: Use this variable to filter patients for which
#          it would be interesting to review whether the
#          computation of si, bac, bsi and sep is right.

# Interesting ids
stay_ids = [
    '200001'
]

rename = {
    'patientunitstayid': 'stay_id',
    'infusionoffset': 'stay_time',
    'labresultoofset': 'stay_time',
    'icustay_id': 'stay_id',
    'admissionid': 'stay_id',
    'start': 'stay_time',
    'startdate': 'stay_time',
    'culturetakenoffset': 'samp',
    'chartdate': 'stay_time'
}

src = 'mimic_0.5.6'

db1 = '%s_raw.parquet' % src
db2 = '%s_cs3t.parquet' % src
db3 = '%s_samp.parquet' % src

# Show information
print("\nDATA: %s" % (DATAPATH / db1))
print("\nLABELS: %s" % (DATAPATH / db2))

# Load data
df1 = pd.read_parquet(DATAPATH / db1)
df2 = pd.read_parquet(DATAPATH / db2)
df3 = pd.read_parquet(DATAPATH / db3)

# Format
df3.rename(columns=rename, inplace=True)

# Format time offsets
df3.stay_time = df3.stay_time / np.timedelta64(1, 'h')


# Save for visual inspection
df2.to_csv('./test.csv')

# Show columns
print("\nColumns (df1):")
print(df1.columns.tolist())
print("\nColumns (df2):")
print(df2.columns.tolist())
print("\nColumns (df3):")
print(df3.columns.tolist())

# Show counts
print("\n\nCounts (df1):")
print(df1.count())
print("\nCounts (df2):")
print(df2.count())
print("\nCounts (df2):")
print(df3.count())

print("Number of samples: %s" % df3.samp.sum())

# Find common ids to visualise
#df1_ids = set(df1.stay_id.unique())
#df2_ids = set(df2.icustay_id.unique())
#common_ids = list(df1_ids.intersection(df2_ids))

#keep_ids = [286720]
#keep_ids = common_ids[0:1]

# Filter
#df1 = df1[df1.stay_id.isin(keep_ids)]
#df2 = df2[df2.icustay_id.isin(keep_ids)]

"""
# Show samples
print("\nSample:")
print(df1[cols])
print("\nSample:")
print(df2)
print("\nSample:")
print(df3)
"""

# Added the samp
merge = df1.merge(df3, how='left',
    left_on=['stay_id', 'stay_time'],
    right_on=['stay_id', 'stay_time'])

# Save
merge.to_parquet(DATAPATH / ('%s_mgd.parquet' % src))

import sys
sys.exit()

# Format
#df2.rename(columns=rename, inplace=True)
df2.samp_time = df2.samp_time / np.timedelta64(1, 'h')
df2.abx_time = df2.abx_time / np.timedelta64(1, 'h')
df2.charttime = df2.charttime / np.timedelta64(1, 'h')


print(df2.shape)
print(df2.icustay_id.nunique())
print(df2.samp_time.notna().count())


aux = df1.merge(df2, how='left',
    left_on=['stay_id', 'stay_time'],
    right_on=['icustay_id', 'charttime'],)

print(aux)

"""
aux = df1.merge(df2[['icustay_id', 'abx_time']],
    left_on=['stay_id', 'stay_time'],
    right_on=['icustay_id', 'abx_time'],
    how='left')


aux = df1.merge(df2[['icustay_id', 'samp_time']],
    left_on=['stay_id', 'stay_time'],
    right_on=['icustay_id', 'samp_time'],
    how='left')

aux = df1.merge(df2,
    left_on=['stay_id', 'stay_time'],
    right_on=['icustay_id', 'charttime'],
    how='left')


def f_baseline_sofa(s):
    if s.isna().all():
        return pd.Series(None, index=s.index, name='bsofa')
    return pd.Series(s - s[s.first_valid_index()], name='bsofa')

aux['bsofa'] = f_baseline_sofa(aux.sofa)

print(aux[cols + ['bsofa', 'delta_sofa', 'charttime', 'samp_time', 'abx_time', 'sep3']])
"""


import sys
sys.exit()

print(df1.dtypes)
print(df2.dtypes)

# Filter interesting ids
# df = df[df.stay_id.isin(stay_ids)]

# See generic info
print("\nSample:")
print(df1)
print("\nColumns:")
print(df1.columns.tolist())
print("\nCounts:")
print(df1.count())


print("\nSample:")
print(df2)
print("\nColumns:")
print(df2.columns.tolist())
print("\nCounts:")
print(df2.count())


#df1.merge(df2, how='left', left_on='stay_time',
#          right_on='')

