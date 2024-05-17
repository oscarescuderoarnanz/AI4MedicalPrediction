# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import datetime
from utils import settings
from utils import display
from utils import prep

#
print(settings.DATAPATH)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
# Select dataset
db = 'eicu_demo_0.5.6.parquet'

features = [
    'hr_raw', 'o2sat_raw', 'temp_raw',
    'sbp_raw', 'dbp_raw', 'map_raw',
    'resp_raw', 'fio2_raw'
]

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
# Read parquet file
df = pd.read_parquet(settings.DATAPATH / db)

count = 0
for id, p in df.groupby(by='stay_id'):
    if count > 5:
        break
    display.plot_patient_stay(p[features].values)

    count += 1

plt.show()

