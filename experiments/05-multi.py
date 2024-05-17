# Libraries
import pandas as pd

from utils import settings
from utils import prep

#
print(settings.DATAPATH)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
# Select dataset
db = 'eicu_demo_0.5.6.parquet'


print(prep.sep(a=2))
print(prep.sepsis(a=2, b=3, c=4))

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
# Read parquet file
#df = pd.read_parquet(settings.DATAPATH / db)

#print(df)


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------