import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def dataframeToTensor(df, timeStepLength, mainKey):
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


def removeIncompleteWindows(data, min_length_patient):
    repetitions = data['stay_id_split'].value_counts()
    idxs = repetitions[repetitions != min_length_patient].index
    idxs_to_remove = data[data['stay_id_split'].isin(idxs)].index
    data = data.drop(idxs_to_remove).reset_index(drop=True)
    return data


def getTrTe(df, seed, debug=True):
    '''
    Rules: 
        - Split the same patients into windows. For them, we define a window length as a parameter.
        - The same patient with all windows can only be included in a single split (either train or test).
        - We will allocate 80% for training and 20% for testing.
        - We won't balance the datasets.
    '''
    # Calculate the minimum number of timeSteps for any patient in the dataset
    min_length_patient = df.groupby("stay_id").size().reset_index()[0].min()

    # Make a copy of the original DataFrame to avoid modifying the original data
    df.stay_id = df.stay_id.astype(int)

    # Assign a sequential count within each group (patient) in the 'stay_id' column to a new column 'row_per_pat'
    df['row_per_pat'] = df.groupby('stay_id').cumcount()
    # Divide patients into groups of 4 rows and assign a 'split' number accordingly
    df['split'] = (df['row_per_pat'] // min_length_patient) + 1

    # Create a new column 'stay_id_split' by concatenating 'stay_id' and 'split' columns
    df['stay_id_split'] = df['stay_id'].astype(str) + '_' + df['split'].astype(str)
    # Drop the auxiliary columns 'row_per_pat' and 'split'
    df = df.drop(['row_per_pat', 'split'], axis=1)
    # Sort the DataFrame by 'stay_id' column
    df_sorted = df.sort_values('stay_id')

    # Group the DataFrame by 'stay_id' and aggregate the values into lists
    df_grouped = df_sorted.groupby('stay_id').agg(lambda x: x.tolist()).reset_index()
    df_grouped.reset_index(drop=True, inplace=True)

    # Split the patients into training and testing sets
    train_stay_ids, test_stay_ids = \
        train_test_split(df_grouped['stay_id'], 
                         test_size=0.2, 
                         random_state=seed)

    # Filter the DataFrame based on the stay_ids in the training and testing sets
    train_df = df[df['stay_id'].isin(train_stay_ids)].reset_index(drop=True)
    train_df = removeIncompleteWindows(train_df, min_length_patient)
    y_train_df = train_df[['stay_id', 'stay_time', 'stay_id_split', 'sepshk_der']].reset_index(drop=True)
    train_df = train_df.drop(['sepshk_der'], axis=1)

    test_df = df[df['stay_id'].isin(test_stay_ids)].reset_index(drop=True)
    test_df = removeIncompleteWindows(test_df, min_length_patient)
    y_test_df = test_df[['stay_id', 'stay_time', 'stay_id_split', 'sepshk_der']].reset_index(drop=True)
    test_df = test_df.drop(['sepshk_der'], axis=1)
    
    if debug:
        print("Length of window:", min_length_patient)
        # Print the number of patients in the training and testing sets
        print("# of windowing patients (train):", len(train_df.stay_id_split.unique()), "- # of original patients:", len(train_df.stay_id.unique()))
        print("# of windowing patients (test):", len(test_df.stay_id_split.unique()), "- # of original patients:", len(test_df.stay_id.unique()))

    return train_df, y_train_df, test_df, y_test_df, min_length_patient