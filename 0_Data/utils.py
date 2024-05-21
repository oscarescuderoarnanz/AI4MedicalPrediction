import pandas as pd
import numpy as np
 
 
def dataframe_to_tensor(df, timeStepLength, mainKey):
    """
    Convert a pandas DataFrame into a numpy tensor for time series analysis.
 
    Args:
    - df: pandas DataFrame, which should contain data for multiple time steps per patient.
    - timeStepLength: Desired length of the time step sequence.
 
    Returns:
    - X: Numpy tensor with dimensions (number of patients, timeStepLength, number of features).
    """
    listPatients = df[mainKey].unique()
    X = []
    for patient in listPatients:
        df_patient = df[df[mainKey] == patient]
        # Drop any rows with missing 'timeStep'
        df_patient = df_patient.dropna(subset=['stay_time'])
        if len(df_patient) >= timeStepLength:
            X_patient = df_patient.iloc[:timeStepLength].values
        else:
            # If the patient has less than timeStepLength time steps, fill with NaNs
            padding = np.full((timeStepLength - len(df_patient), df.shape[1]), np.nan)
            X_patient = np.vstack([df_patient.values, padding])
        X.append(X_patient)
 
    return np.array(X, dtype="float")
 
def LVCF(df):
    df = df.sort_values(by=['stay_id', 'stay_time'])
    for patient in df['stay_id'].unique():
        last_value = df[df['stay_id'] == patient].fillna(method='ffill').iloc[-1]
        df.loc[df['stay_id'] == patient, df.columns[2:]] = df.loc[df['stay_id'] == patient, df.columns[2:]].fillna(last_value)
    return df
 
 
def slidingWindow(df, moving_span, window_length):
 
    list_patients = np.unique(df.stay_id)
    df_sw_ttl = pd.DataFrame()
    for idx_pat in range(len(list_patients)):
        df_sw = pd.DataFrame()
        df_patient = df[df.stay_id == list_patients[idx_pat]]
        id_pat = df_patient.stay_id.unique()[0]
        iterations = int(np.ceil((df_patient.shape[0] - (window_length)+1) / moving_span))
        for j in range(iterations):
            df_aux = df_patient[moving_span*j: (window_length + moving_span*j)]
            df_aux['w_id'] = str(id_pat) + "_" + str(j)
            df_sw = pd.concat([df_sw, df_aux],ignore_index=True)
 
        df_sw_ttl = pd.concat([df_sw_ttl, df_sw],ignore_index=True)
 
    return df_sw_ttl
 

def filter_windows(df, N, M):
    w = []
    
    for stay_id, group in df.groupby('stay_id'):
        group = group.reset_index(drop=True)
        sep_onset_indices = group[group['sep_onset'] == 1].index.tolist()
        
        # Obtener todos los w_id únicos en el grupo
        unique_w_ids = group['w_id'].unique()
        
        for idx in sep_onset_indices:
            onset_w_id = group.loc[idx, 'w_id']
            
            # Extraer la parte numérica del w_id para la comparación
            onset_w_id_num = int(onset_w_id.split('_')[1])
            
            # Calcular los rangos de w_id
            start_w_id_num = onset_w_id_num - N
            end_w_id_num = onset_w_id_num + M
            
            # Filtrar los w_id que están en el rango deseado
            valid_w_ids = [f"{onset_w_id.split('_')[0]}_{i}" for i in range(start_w_id_num, end_w_id_num + 1)]
            
            # Filtrar las filas en el rango de w_id
            filtered_group = group[group['w_id'].isin(valid_w_ids)]
            w.append(filtered_group)
    
    return pd.concat(w, ignore_index=True)



def filter_2w_per_patient(df_final):
    pats = df_final[df_final.sep_onset == 1].stay_id.unique()
    df_aux = df_final[df_final.stay_id.isin(pats)].reset_index(drop=True)
    df_final_filter = pd.DataFrame(columns=df_aux.columns)
    for i in range(len(pats)):
        df_pat = df_aux[df_aux.stay_id == pats[i]]
        df_pat = df_pat.sort_values(by=['stay_id', 'w_id'])

        # Get the firts window (min w_id) of each patient
        first_window = df_pat.groupby('stay_id')['w_id'].min().reset_index()
        first_window = first_window.rename(columns={'w_id': 'first_w_id'})

        # Union datasets
        df_pat = pd.merge(df_pat, first_window, on='stay_id')

        try:
            first_window_df_pat = df_pat[df_pat['w_id'] == df_pat['first_w_id']]
            onset_sep_df_pat = df_pat[df_pat['sep_onset'] == 1]
            arr = onset_sep_df_pat.w_id.values
            split_values = [(val, int(val.split('_')[1])) for val in arr]
            max_value = max(split_values, key=lambda x: x[1])
            result = max_value[0]
            onset_sep_df_pat = df_pat[df_pat['w_id'] == result]
        except:
            onset_sep_df_pat = pd.DataFrame(columns=df_pat.columns)
            
        # Combine filters to obtain the final result
        result_df_pat = pd.concat([first_window_df_pat, onset_sep_df_pat]).drop_duplicates()
        result_df_pat = result_df_pat.reset_index(drop=True)
        result_df_pat = result_df_pat.drop(['first_w_id'], axis=1)
        df_final_filter = pd.concat([df_final_filter, result_df_pat], axis=0, ignore_index=True)

    df_final_filter = df_final_filter.reset_index(drop=True)
    
    return df_final_filter
 
 
def get_SI(df, Nts_pre, Nts_post):
    df['SI'] = 0
    df['abx'] = df['abx'].fillna(False)
    pats = list(df.stay_id.unique())
    for i in range(len(pats)):
        data = df[df.stay_id == pats[i]]
        try:
            true_index = data[data['abx'] == True].index[0]
            start_index = max(0, true_index - Nts_pre)
            end_index = min(len(df), true_index + Nts_post + 1)
            df.loc[start_index:end_index, 'SI'] = 1
        except IndexError:
            continue
    return df
 
def prop_sep(df, n_prog):
    df['sep_'+str(n_prog)] = 0
    for pat, grupo in df.groupby('stay_id'):
        sep_onset_indices = grupo.index[grupo['sep_onset'] == 1]
        for index in sep_onset_indices:
            df.loc[index:index+n_prog, 'sep_'+str(n_prog)] = 1
    return df
 
 
def get_sep(df, N_prog_sep, increm):
 
    df['diff'] = df.groupby('stay_id')['sofa'].diff()
    df['sep_onset'] = ((df['diff'] >= increm) & (df['SI'] == True)).astype(int)
    df = prop_sep(df, N_prog_sep)
    return df

