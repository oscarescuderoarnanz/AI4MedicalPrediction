import pandas as pd
import numpy as np
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")

def eicu_micro(path1, path2):

    df = pd.read_csv(path1)
    df['stay_time'] = df['culturetakenoffset'].apply(lambda x: round(x / 60))

    df = df.sort_values(by=['patientunitstayid', 'stay_time'])

    df_pivot = df.pivot_table(index=['patientunitstayid', 'stay_time'],
                              columns='organism', 
                              aggfunc='size')


    df_micro = df_pivot.reset_index()
    df_micro = df_micro.rename(columns={'patientunitstayid':'stay_id'}, inplace=False)

    micro_pats = df_micro.stay_id.unique()
    print("# of micro pats:", len(micro_pats))

    df = pd.read_parquet(path2)

    ttl_pats = df.stay_id.unique()
    print("# of pats:", len(ttl_pats))

    df_merge = pd.merge(df, df_micro, how="left", on=["stay_id", "stay_time"])

    df_merge.to_parquet('df_eicu_with_micro.parquet',engine='pyarrow')
    
   
    
    
if __name__ == '__main__':
    
    path1 = "../datasets/raw_data/microLab.csv"
    path2 = '../datasets/eicu_0.5.6.parquet'
    eicu_micro(path1, path2)