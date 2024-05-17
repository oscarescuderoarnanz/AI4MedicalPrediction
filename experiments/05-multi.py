# Libraries
import pandas as pd

# Own
import settings
import utils

#
print(settings.DATAPATH)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
# Select dataset
db = 'eicu_demo_0.5.6.parquet'


print(utils.sep(a=2))
print(utils.sepsis(a=2, b=3, c=4))

# ----------------------------------------------------
# Load data
# ----------------------------------------------------
# Read parquet file
#df = pd.read_parquet(settings.DATAPATH / db)

#print(df)


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------