import pandas as pd
import numpy as np
import time

import warnings
warnings.filterwarnings("ignore")

import pyten
from sklearn.preprocessing import StandardScaler

import utils

def getSplits(path, imputationType, thr_nan, seed, debug=False):

    # Load data
    df = pd.read_parquet(path)

    # Step -1. Select data of ICU
    df_icu_entry = df[df.stay_time >= 0].reset_index(drop=True)

    if debug:
        print("% of missing values (pre-filter):", np.round((df.isnull().sum().sum() / \
                                                             (df.shape[0]*df.shape[1])*100), 4))
        print("% of missing values label:", df.sepshk_der.isnull().sum())
        print("% of missing values (post-filter):", np.round((df_icu_entry.isnull().sum().sum() / \
                                                              (df_icu_entry.shape[0]*df_icu_entry.shape[1])*100), 4))
        print("% of missing values label:", df_icu_entry.sepshk_der.isnull().sum(), "\n")

    # Setp 0. Filter by label. Remove patients with missing data in label 
    nans = df_icu_entry['sepshk_der'].isna()
    idxs = df_icu_entry.index[~nans]
    df_final = df_icu_entry.loc[idxs]

    if debug:
        print("# of patients pre-filter:", len(df_icu_entry.stay_id.unique()), "- shape:", df_icu_entry.shape)
        print("Values missing values label:", df_icu_entry.sepshk_der.isnull().sum())
        print("# of patients post-filter:", len(df_final.stay_id.unique()), "- shape:", df_final.shape)
        print("Values missing values label:", df_final.sepshk_der.isnull().sum(), "\n")

    ## Step 1. Select relevant features. Remove feature without data, based on threshold.
    missing_percentage = df_final.isnull().mean() * 100
    feats = missing_percentage[missing_percentage > thr_nan].index.tolist()

    if debug:
        print("Features with more than " + str(thr_nan) + "% missing data:" +  str(len(feats)))

    df_filter = df_final.drop(feats, axis=1)
    df_final = df_filter.astype(float)

    if imputationType == "LVCF":
        df_final = utils.LVCF(df_final)
        df_final = df_final.fillna(0)

    train_df, y_train_df, test_df, y_test_df, length_window = utils.getTrTe(df_final, seed)

    # Normalization
    scaler = StandardScaler()
    scaler.fit(train_df[list(train_df.keys())[2:-1]])
    X_train_scaled = scaler.transform(train_df[list(train_df.keys())[2:-1]])
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=train_df.columns[2:-1])
    X_train_scaled = pd.concat([X_train_scaled, train_df[['stay_id', 'stay_time', 'stay_id_split']]], axis=1)

    X_test_scaled = scaler.transform(test_df[list(test_df.keys())[2:-1]])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=test_df.columns[2:-1])
    X_test_scaled = pd.concat([X_test_scaled, test_df[['stay_id', 'stay_time', 'stay_id_split']]], axis=1)

    # Convert the DataFrames into tensors while discarding the first two and last features
    X_train = utils.dataframeToTensor(X_train_scaled, length_window, 'stay_id_split')
    X_train = X_train[:, :, 0:X_train.shape[2]-3]
    X_test = utils.dataframeToTensor(X_test_scaled, length_window, 'stay_id_split')
    X_test = X_test[:, :, 0:X_test.shape[2]-3]
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)
    
    return X_train, X_test, y_train_df, y_test_df