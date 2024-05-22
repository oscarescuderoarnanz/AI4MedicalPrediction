import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
 
import pyten
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
import utils
 
 
def preprocessing(params, sep_def, debug=False):
 
    # Load data
    df = pd.read_parquet(params['path'])
    print("# of patients:", len(df.stay_id.unique()))
    df = utils.get_SI(df, sep_def['Nts_pre'], sep_def['Nts_post'])
    df = utils.get_sep(df, sep_def['N_prog_sep'],  sep_def['increm_sofa'])
    df = df.drop(['diff'], axis=1)
    
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
    df_final = utils.slidingWindow(df_final, params['moving_span'], min_length_pat)
    if debug:
        print("Dimensiones post-sliding window:", df_final.shape)
        
    return df_final
    
 
 
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
 
    # Convert the DataFrames into tensors while discarding the first two and last features
    X_train = utils.dataframe_to_tensor(X_train_scaled, params['min_length_pat'], 'w_id')
    X_train = X_train[:, :, 0:X_train.shape[2]-3]
    X_test = utils.dataframe_to_tensor(X_test_scaled, params['min_length_pat'], 'w_id')
    X_test = X_test[:, :, 0:X_test.shape[2]-3]
 
    y_train = utils.dataframe_to_tensor(y_train_df, params['min_length_pat'], 'w_id')
    y_train = y_train[:, :, -1]
    y_test = utils.dataframe_to_tensor(y_test_df, params['min_length_pat'], 'w_id')
    y_test = y_test[:, :, -1]
    return  X_train, X_test, y_train, y_test, train_df.columns

