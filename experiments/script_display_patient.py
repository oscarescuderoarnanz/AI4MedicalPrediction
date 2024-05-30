# Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

#from ... import display
from utils.settings import DATAPATH
from utils import display

# Show information
print(DATAPATH)

# --------------------------------------
# Configuration
# --------------------------------------
# Define output path
OUTPATH = Path('./outputs/patients/')

from collections import OrderedDict


d = OrderedDict({
    'stay_id': {},
    'stay_time': { 'cmap': 'Greens' },
    'abx': { 'cmap': 'coolwarm' },
    'samp': { 'cmap': 'coolwarm' },
    'si_ons_moor2023': { 'cmap': 'coolwarm' },
    'sep_on_moor2023': { 'cmap': 'coolwarm' },
    'sofa': { 'cmap': 'Blues' },
    'bsofa': { 'cmap': 'Blues' },
    'dsofa': { 'cmap': 'Blues' },
    'hr': {},
    'o2sat': {},
    'temp': {},
    'sbp': {},
    'dbp': {},
    'map': {},
    'resp': {},
    'fio2': {},
    'po2': {},
    'bili': {},
    'plt': {},
    'crea':{},
    'sirs': { 'cmap': 'Blues' },
    'news': { 'cmap': 'Blues' },
    'mews': { 'cmap': 'Blues' }
})

colormaps = [
    v.get('cmap', 'Reds') for k,v in d.items()
][1:]

# Redraw images
RESET = False


# --------------------------------------
# Load data
# --------------------------------------
# Define source
src = 'mimic_demo'

# Select data set
db = '%s_0.5.6_lbl.parquet' % src

# Show information
df = pd.read_parquet(DATAPATH / db)
df = df[d.keys()]
df = df.convert_dtypes(convert_boolean=False)

# .. note: This is to filter the patients that we we want to be
#          displayed. It can be done according to their stay_id
#          or whether they have suspicion of infection,
#          bacteremia or sepsis.

# Filter patients
keep_ids = []
if len(keep_ids):
    df = df[df.stay_id.isin(keep_ids)]

# Show
print(df)

# Loop and display
groups = df.groupby(by='stay_id', as_index=False)
count = 0
for i, (stay_id, p) in enumerate(groups) :
    # Log
    print("%s/%s. Displaying... %s" % (i, len(groups), stay_id))

    # Skip for specific reason
    if p.samp.isna().all() or p.abx.isna().all():
        continue

    #if i>1:
    #    break

    # Format matrix
    m = p.drop(columns=['stay_id'])
    m.abx.fillna(0, inplace=True)
    m = m.convert_dtypes() #.fillna(0)

    # Display
    f, ax = display.plot_patient_stay(
        m=m.to_numpy(dtype=np.float32),
        xticklabels=m.stay_time.astype(int).tolist(),
        yticklabels=m.columns.tolist(),
        colormaps=colormaps
        #label=stay_id
    )

    count += 1

    # Show
    plt.savefig(OUTPATH / ('%s' % src) / ('%s_%s.pdf' % (src, stay_id)))



import sys
sys.exit()

"""
# Display
for i, g in df.groupby(by='stay_id', as_index=False):
    aux = g.set_index(['stay_id', 'stay_time']) \
        .stack().reset_index()
    aux.columns = ['stay_id', 'stay_time', 'code', 'value']
    aux.to_json(output / ('%s.json' % i),
        orient='records', indent=4)

    g.to_csv(output/ ('%s.csv' % i))

import sys
sys.exit()

groups = df.groupby(by='stay_id', as_index=False)
count = 0
for i, (stay_id, p) in enumerate(groups):

    # Log
    print("%s/%s. Displaying... %s" % (i, len(groups), stay_id))


    if p.samp.isna().all() or p.abx.isna().all():
        continue

    # Stop
    if count > 2:
        break

    print("a")

    # Format matrix
    m = p.drop(columns=['stay_id'])
    m.abx.fillna(0, inplace=True)
    m = m.convert_dtypes() #.fillna(0)

    print("b")
    # Display
    f, ax = display.plot_patient_stay(
        m=m.to_numpy(dtype=np.float32),
        xticklabels=m.stay_time.astype(int).tolist(),
        yticklabels=m.columns.tolist(),
        colormaps=['Greens'] + ['coolwarm']*4,
        blabel=stay_id
    )

    count += 1

    # Show
    plt.show()
"""