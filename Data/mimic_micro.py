import pandas as pd
import numpy as np
from datetime import timedelta

import warnings
warnings.filterwarnings("ignore")


   
def mimic_micro(path1, path3, path2):
    df_mimic = pd.read_parquet(path2)

    df_micro = pd.read_csv(path1)
    df_icu= pd.read_csv(path2)

    pats_mimic = list(df_mimic.stay_id.unique())
    df_icu = df_icu[df_icu.stay_id.isin(pats_mimic)].reset_index(drop=True)

    # Select some relevant variables
    df_micro_1 = df_micro[['subject_id', 'hadm_id', 'test_name', 'org_name', 'charttime']]
    # Transform upper to lower 
    df_micro_1['test_name'] = df_micro_1['test_name'].str.lower()
    # filter data
    df_micro_2 = df_micro_1[df_micro_1['test_name'].str.contains('blood culture')].reset_index(drop=True)

    # Merge icu patients with micro
    print("SHAPE df_icu", df_icu.shape)
    print("SHAPE df_micro_2", df_micro_2.shape)
    df_micro_2 = df_micro_2.drop(['hadm_id'], axis=1)
    df_merge = pd.merge(df_icu[['intime', 'outtime', 'subject_id', 'stay_id']], df_micro_2, how="left", on=["subject_id"])

    # Time (minutes) from icu entry to culture date
    df_merge['intime'] = pd.to_datetime(df_merge['intime'])
    df_merge['charttime'] = pd.to_datetime(df_merge['charttime'])
    df_merge['offset'] = (df_merge['charttime'] - df_merge['intime']).dt.total_seconds() / 60
    # Remove samples without offset
    df_merge = df_merge.dropna(subset=['offset'])
    # Create stay_time feature
    df_merge['stay_time'] = df_merge['offset'].apply(lambda x: round(x / 60))

    df_merge = df_merge.sort_values(by=['stay_id', 'stay_time'])
    df_pivot = df_merge.pivot_table(index=['stay_id', 'stay_time'],
                              columns='org_name', 
                              aggfunc='size')


    df_mimic_micro = df_pivot.reset_index()
    df_mimic_micro_merge = pd.merge(df_mimic, df_mimic_micro, how="left", on=["stay_id", "stay_time"])
    
    df_mimic_micro_merge.to_parquet('df_mimic_with_micro.parquet',engine='pyarrow')

    
    
if __name__ == '__main__':
    
    path1 = "../datasets/raw_data/mimic-iv-2.2/hosp/microbiologyevents.csv"
    path2 = "../datasets/raw_data/mimic-iv-2.2/icu/icustays.csv"
    path3 = '../datasets/miiv_0.5.6.parquet'
    mimic_micro(path1, path3, path2)
