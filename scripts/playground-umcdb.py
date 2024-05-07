# Libraries
import pandas as pd

from pathlib import Path

"""
.. notes:

Possible features:

admissions.cvs
   - admissioncount (or re-admission)
    

"""

# ---------------------------------------------
# Configuration
# ---------------------------------------------
# Path
PATH = Path('../datasets/amsterdam-umcdb')

# Load
df_adm = pd.read_csv(PATH / 'admissions.csv', nrows=100)
df_drg = pd.read_csv(PATH / 'drugitems.csv', nrows=100)
df_lst = pd.read_csv(PATH / 'listitems.csv', nrows=100)
df_prc = pd.read_csv(PATH / 'processitems.csv', nrows=100)
df_prc_order = pd.read_csv(PATH / 'procedureorderitems.csv', nrows=100)

print(df_adm)
#print(df_drg)
#print(df_lst)

#