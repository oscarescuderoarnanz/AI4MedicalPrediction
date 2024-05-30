import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
 
import pyten
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
import utils
from functools import partial




############################
#### General functions #####
############################
def preprocessing(params, sep_def, debug=False):
 
    # Load data
    df = pd.read_parquet(params['path'])
    print("# of patients:", len(df.stay_id.unique()))
    df = utils.get_SI(df, sep_def['Nts_pre'], sep_def['Nts_post'])
    if sep_def['ref_sofa_icu']:
        df = df[df.stay_time >= 0].reset_index(drop=True)
    df['bsofa'] = df.groupby('stay_id')['sofa'].apply(utils.f_baseline_sofa).reset_index(level=0, drop=True)
    df = utils.get_sep(df, sep_def['N_prog_sep'], sep_def['increm_sofa'])
    df = df.drop(["bsofa"], axis=1)
    
    if params["filter_pat"]:
        pats = df[df.sep_onset == 1].stay_id.unique()
        df = df[df.stay_id.isin(pats)]

    # Step -1. Select data of ICU
    df_icu_entry = df[df.stay_time >= 0].reset_index(drop=True)
    print("# of icu-patients:", len(df_icu_entry.stay_id.unique()))

    if debug:
        print("% of missing values (pre-filter):", np.round((df.isnull().sum().sum() / \
                                                             (df.shape[0]*df.shape[1])*100), 4))
        print("% of missing values (post-filter):", np.round((df_icu_entry.isnull().sum().sum() / \
                                                              (df_icu_entry.shape[0]*df_icu_entry.shape[1])*100), 4))
        print("# of patients labels:", len(df_icu_entry.stay_id.unique()))
        print("% of missing values post remove patients with nan label:", np.round((df_icu_entry.isnull().sum().sum() / \
                                                             (df_icu_entry.shape[0]*df_icu_entry.shape[1])*100), 4))



    ## Step 1. Select relevant features. Remove feature without data, based on threshold.
    df_final = df_icu_entry[params['keys']]

    ## Step 2. Filter patients in based on theirs information
    if params["filter_patients_nans"]:
        counts = df_final.groupby('stay_id').count()
        total_values = df_final.groupby('stay_id').size()
        percentages_all_features = counts.div(total_values, axis=0)

        counts = df_final.groupby('stay_id').count().sum(axis=1)
        total_values = df_final.groupby('stay_id').size()  * df_final.shape[1]
        percentages = (counts / total_values) * 100
        percentages.name = 'total_percentage'
        percentage_values = pd.DataFrame(percentages).reset_index()

        df_per_with_values = pd.merge(percentages_all_features, percentage_values, how="left", on="stay_id")
        print("# of patients pre filter by information:", len(df_per_with_values.stay_id.unique()))
        pats = df_per_with_values[df_per_with_values.total_percentage > params["th"]].stay_id.unique()
        print("# of patients post filter by information:", len(pats))
        df_final = df_final[df_final.stay_id.isin(pats)]
    
    if debug:
        
        keys_to_anal = ['sofa', 'hr_raw', 'o2sat_raw', 'temp_raw', 'sbp_raw',
                  'dbp_raw', 'map_raw', 'resp_raw', 'fio2_raw', 'po2_raw',
                  'bili_raw', 'plt_raw', 'crea_raw', 'sirs', 'news', 'mews',
                 'SI']
            
        values = df_per_with_values[keys_to_anal].median(axis=0).values
        keys = list(df_per_with_values[keys_to_anal].median(axis=0).keys())

        plt.figure(figsize=(8,4))
        plt.bar(keys, values)
        plt.xticks(rotation=90)
        plt.grid()
        plt.show()

        print("Dimensions post remove some feautures:", df_final.shape)
 
    if params['imputationType'] == "LVCF":
        df_final = utils.LVCF(df_final)
        df_final = df_final.fillna(0)
        print("# of patients post imputation:", len(df_final.stay_id.unique()))
    
    if params['min_length_pat'] == 0:
        min_length_pat = df_final.groupby("stay_id").size().reset_index()[0].min()
    
    params['min_length_pat'] = min_length_pat
    df_final.stay_id = df_final.stay_id.astype(int)
 
    if debug:
        print("# of patients:", len(df_final.stay_id.unique()))
        print("# of pacientes labeled with 1", len(df_final[df_final.sep_onset == 1][['stay_id']].stay_id.unique()))
        print("# of pacientes labeled with 0", len(df_final[df_final.sep_onset == 0][['stay_id']].stay_id.unique()))
        print("Dimensiones pre-sliding window:", df_final.shape)
        
    return df_final, min_length_pat

############################
# Functions for approach 1 #
############################
def filter_group(grupo, lw):
    indices_seponset_1 = grupo.index[grupo['sep_onset'] == 1]
    if len(indices_seponset_1) > 0:
        return all((y - x) >= lw for x, y in zip(indices_seponset_1[:-1], indices_seponset_1[1:]))
    else:
        return False
    
def processing_first_approach(df, params):
    
    length_window = params["length_window"]
    df_to_work = df.reset_index(drop=True)

    # Group by 'stay_id' and sum the values
    df_nosepsis = df_to_work.groupby(by="stay_id").sum().reset_index()
    # Identify patients with no sepsis
    pats_no_sepsis = df_nosepsis[df_nosepsis['sep_onset'] == 0].stay_id.unique()
    # Filter the original df_sepsisframe to include only patients with no sepsis
    df_no_sepsis = df_to_work[df_to_work['stay_id'].isin(pats_no_sepsis)].reset_index(drop=True)
    print("# of patients without sepsis:", len(df_no_sepsis.stay_id.unique()), "-", df_no_sepsis.shape)
    df_sepsis = df_to_work[~df_to_work['stay_id'].isin(pats_no_sepsis)].reset_index(drop=True)
    print("# of patients with sepsis:", len(df_sepsis.stay_id.unique()), "-", df_sepsis.shape)


    # NO SEPSIS PATIENTS - PREPROCESSING
    df_no_sepsis_filtered = df_no_sepsis[df_no_sepsis['stay_time'] >= (length_window-1)].stay_id.unique()
    df_no_sepsis = df_no_sepsis[df_no_sepsis.stay_id.isin(df_no_sepsis_filtered)].reset_index()
    df_no_sepsis = df_no_sepsis.groupby('stay_id').head(7).reset_index(drop=True).drop(['index'],axis=1)
    print("shape pre filter:", df_no_sepsis.shape)
    filt = lambda x: len(x) >= params["length_window"]
    df_no_sepsis = df_no_sepsis.groupby('stay_id').filter(filt)
    print("shape post filter:", df_no_sepsis.shape)

    # SEPSIS PATIENTS - PREPROCESSING
    df_sepsis_filt = df_sepsis[df_sepsis['sep_onset'] == 1]
    df_sepsis_filt = df_sepsis_filt.sort_values(by=['stay_id', 'stay_time'])
    df_sepsis_filt = df_sepsis_filt.drop_duplicates(subset='stay_id', keep='first')

    filtered_patients = df_sepsis_filt[(df_sepsis_filt['stay_time'] >= (length_window-1)) & (df_sepsis['sep_onset'] == 1)]
    pats_with_condition = filtered_patients['stay_id'].unique()
    df_sepsis = df_sepsis[df_sepsis['stay_id'].isin(pats_with_condition)].reset_index(drop=True)

    rows = []

    for stay_id, grupo in df_sepsis.groupby('stay_id'):
        idx_seponset = grupo.index[grupo['sep_onset'] == 1]

        if not idx_seponset.empty:
            idx_init = max(0, idx_seponset[0] - (length_window - 1))
            selected_rows = grupo.loc[idx_init:idx_seponset[0]]

            if len(selected_rows) == length_window:
                rows.append(selected_rows)

    df_sepsis_final = pd.concat(rows, ignore_index=True)

    print("shape final:", df_sepsis_final.shape)
    assert df_sepsis_final.shape[0] % length_window == 0, "Error: No todos los pacientes tienen exactamente 7 filas."
    print("shape post filter:", df_sepsis.shape)
    print()
    print("# of patients with sepsis: ", df_sepsis_final.shape[0]/length_window)
    print("# of patients without sepsis: ", df_no_sepsis.shape[0]/length_window)

    df_concat = pd.concat([df_no_sepsis, df_sepsis_final], axis=0)
    
    return df_concat
 
 

def get_tr_te_firstapp(df, params, seed, debug=True):
    '''
    Rules: 
        - Split the same patients into windows. For them, we define a window length as a parameter.
        - The same patient with all windows can only be included in a single split (either train or test).
        - We will allocate 80% for training and 20% for testing.
        - We won't balance the datasets.
    '''
    # Split the patients into training and testing sets
    # Split the patients into training and testing sets
    train_stay_ids, test_stay_ids = \
        train_test_split(df.stay_id.unique(), 
                         test_size=0.2, 
                         random_state=seed)

    # Filter the DataFrame based on the stay_ids in the training and testing sets
    train_df = df[df['stay_id'].isin(train_stay_ids)].reset_index(drop=True)
    y_train_df = train_df[params['f_tr_te']].reset_index(drop=True)
    train_df = train_df.drop(params['label'], axis=1)

    test_df = df[df['stay_id'].isin(test_stay_ids)].reset_index(drop=True)
    y_test_df = test_df[params['f_tr_te']].reset_index(drop=True)
    test_df = test_df.drop(params['label'], axis=1)

    # Normalization
    scaler = StandardScaler()
    scaler.fit(train_df[list(train_df.keys())[2:]])
    X_train_scaled = scaler.transform(train_df[list(train_df.keys())[2:]])
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=train_df.columns[2:])
    X_train_scaled = pd.concat([X_train_scaled, train_df[['stay_id', 'stay_time']]], axis=1)

    X_test_scaled = scaler.transform(test_df[list(test_df.keys())[2:]])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=test_df.columns[2:])
    X_test_scaled = pd.concat([X_test_scaled, test_df[['stay_id', 'stay_time']]], axis=1)

    # # Convert the DataFrames into tensors while discarding the first two and last features
    X_train = utils.dataframe_to_tensor(X_train_scaled, params['length_window'], "stay_id")
    X_train = X_train[:, :, 0:X_train.shape[2]-2]
    X_test = utils.dataframe_to_tensor(X_test_scaled, params['length_window'], "stay_id")
    X_test = X_test[:, :, 0:X_test.shape[2]-2]

    X_train = X_train[:,0:params['length_window']-1]
    X_test = X_test[:,0:params['length_window']-1]

    y_train = y_train_df.groupby('stay_id').tail(1).reset_index(drop=True).sep_onset.values
    y_test = y_test_df.groupby('stay_id').tail(1).reset_index(drop=True).sep_onset.values

    return  X_train, X_test, y_train, y_test, train_df.columns




############################
# FUNCTIONS FOR APPROACH 2 #
############################

def get_tr_te(df, params, seed, debug=True):
    '''
    Rules: 
        - Split the same patients into windows. For them, we define a window length as a parameter.
        - The same patient with all windows can only be included in a single split (either train or test).
        - We will allocate 80% for training and 20% for testing.
        - We won't balance the datasets.
    '''
 
    # Split the patients into training and testing sets
    train_stay_ids, test_stay_ids = \
        train_test_split(df.stay_id.unique(), 
                         test_size=0.2, 
                         random_state=seed)
 
    # Filter the DataFrame based on the stay_ids in the training and testing sets
    train_df = df[df['stay_id'].isin(train_stay_ids)].reset_index(drop=True)
    y_train_df = train_df[params['f_tr_te']].reset_index(drop=True)
    train_df = train_df.drop(params['label'], axis=1)
 
    test_df = df[df['stay_id'].isin(test_stay_ids)].reset_index(drop=True)
    y_test_df = test_df[params['f_tr_te']].reset_index(drop=True)
    test_df = test_df.drop(params['label'], axis=1)

 
    if debug:
        # Print the number of patients in the training and testing sets
        print("# of windowing patients (train):", len(train_df.w_id.unique()), "- # of original patients:", len(train_df.stay_id.unique()))
        print("# of windowing patients (test):", len(test_df.w_id.unique()), "- # of original patients:", len(test_df.stay_id.unique()))
    # Normalization
    scaler = StandardScaler()
    scaler.fit(train_df[list(train_df.keys())[2:-1]])
    X_train_scaled = scaler.transform(train_df[list(train_df.keys())[2:-1]])
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=train_df.columns[2:-1])
    X_train_scaled = pd.concat([X_train_scaled, train_df[['stay_id', 'stay_time', 'w_id']]], axis=1)
 
    X_test_scaled = scaler.transform(test_df[list(test_df.keys())[2:-1]])
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=test_df.columns[2:-1])
    X_test_scaled = pd.concat([X_test_scaled, test_df[['stay_id', 'stay_time', 'w_id']]], axis=1)
    
    w_id_tr = X_train_scaled[['stay_id', 'stay_time', 'w_id']]
    w_id_te = X_test_scaled[['stay_id', 'stay_time', 'w_id']]
 
    # Convert the DataFrames into tensors while discarding the first two and last features
    X_train = utils.dataframe_to_tensor(X_train_scaled, params['min_length_pat'], 'w_id')
    X_train = X_train[:, :, 0:X_train.shape[2]-3]
    X_test = utils.dataframe_to_tensor(X_test_scaled, params['min_length_pat'], 'w_id')
    X_test = X_test[:, :, 0:X_test.shape[2]-3]
 
    y_train = utils.dataframe_to_tensor(y_train_df, params['min_length_pat'], 'w_id')
    y_train = y_train[:, :, -3]
    y_test = utils.dataframe_to_tensor(y_test_df, params['min_length_pat'], 'w_id')
    y_test = y_test[:, :, -3]
    return  X_train, X_test, y_train, y_test, train_df.columns, w_id_tr, w_id_te

