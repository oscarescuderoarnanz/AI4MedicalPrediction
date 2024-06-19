# Libraries
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------
# Methods
# ---------------------------------------------------------------------
def extract_window_metadata(group, look_back=6, look_ahead=3, step=1):
    """Function to define window metadata for a single patient.

    .. note:: The step functionality is not being used yet.

    Parameters
    ----------
    group: pd.DataFrame
        The DataFrame with the information for a single patient. It is required
        to have a column named <stay_time> which indicates the time (hour, minute,
        or day) within the patient data timeline.
    look_back: int
        The number of time steps before current time to keep.
    look_ahead: int
        The number of time steps after current_time to keep.
    step: int
        The number of time steps between consecutive windows.

    Returns
    -------
    DataFrame with the windows metadata
    """
    # Initialize a list to hold window metadata for the current group
    window_metadata = []

    # Iterate over each row in the group
    for i in range(0, len(group), step):
        # Current row's stay_time
        current_stay_time = group.iloc[i]['stay_time']

        # Calculate the time window bounds
        start_time = current_stay_time - look_back
        end_time = current_stay_time + look_ahead

        # Find indices within the time window
        # Using boolean indexing to find rows within the window
        window_indices = group[(group['stay_time'] >= start_time) &
                               (group['stay_time'] <= end_time)].index

        # Find the first and last index in this window
        start_idx = window_indices[0] if len(window_indices) > 0 else i
        end_idx = window_indices[-1] if len(window_indices) > 0 else i

        # .. note: the stay_id is the same for all the rows.
        # .. note: unique window within each stay_id.
        # Get stay id
        stay_id = group['stay_id'].iloc[0]

        # Append the metadata for the window
        window_metadata.append({
            'stay_id': stay_id,
            'window_id': '%s_%s' % (stay_id, i // step + 1),
            'current_stay_time': current_stay_time,
            'start_index': start_idx,
            'end_index': end_idx,
            #'is_sep_onset': group.loc[start_idx:end_idx, 'sep_on_moor2023'].sum() > 0
        })

    # Convert the list of dictionaries to a DataFrame
    return pd.DataFrame(window_metadata)


def extract_window_data(original_data, window_metadata):
    """Function to extract actual data using window metadata.

    Parameters
    ----------
    original_data: pd.DataFrame
        The original DataFrame containing all data.
    window_metadata: pd.DataFrame
        The window metadata DataFrame. It must include the following
        columns <stay_id>, <start_index>, and <end_index>.

    Returns
    -------
    Original DataFrame windows concatenated
    """
    extracted_data = []

    for idx, window in tqdm(window_metadata.iterrows(), total=window_metadata.shape[0]):
        stay_id = window['stay_id']
        start_idx = window['start_index']
        end_idx = window['end_index']

        # Extract data for this window
        window_data = original_data[(original_data['stay_id'] == stay_id) &
                                    (original_data.index >= start_idx) &
                                    (original_data.index <= end_idx)].copy()

        # Add window_id and initial_stay_time to the extracted data
        window_data['window_id'] = window['window_id']
        window_data['current_stay_time'] = window['current_stay_time']

        extracted_data.append(window_data)

    # Concatenate all extracted data into a single DataFrame
    extracted_data_df = pd.concat(extracted_data, ignore_index=True)

    # Return
    return extracted_data_df


def generate_window_metadata(df, look_back, look_ahead, step=1):
    """Function to generate window metadata.

    .. warning:: There is a warning that needs to be solved.
              window.py:131: DeprecationWarning: DataFrameGroupBy.apply operated
              on the grouping columns. This behavior is deprecated, and in a future
              version of pandas the grouping columns will be excluded from the
              operation. Either pass `include_groups=False` to exclude the groupings
              or explicitly select the grouping columns after groupby to silence
              this warning.

    Parameters
    ----------
    df: pd.DataFrame
        The DataFrame with the information. It is required to have a column
        <stay_id> to identify individual patients/stays and another column
        named <stay_time> which indicates the time within the patient data
        timeline (e.g. hour, day, ...).
    look_back: int
        The number of time steps before current time to keep.
    look_ahead: int
        The number of time steps after current_time to keep.
    step: int
        The number of time steps between consecutive windows.

    Returns
    -------
    DataFrame with the windows metadata

    """
    # Ensure the DataFrame is sorted by stay_id and stay_time
    df = df.sort_values(by=['stay_id', 'stay_time'])

    # Apply the extract_window_metadata function to each group of stay_id
    windows_metadata = df.groupby('stay_id', group_keys=False).progress_apply(
        extract_window_metadata, look_back=look_back,
        look_ahead=look_ahead, step=step
    ).reset_index(drop=True)  # Flatten the DataFrame

    # Filter out any windows that do not have complete data
    expected_rows = look_back + look_ahead + 1  # +1 includes the current stay_time
    complete_windows_metadata = windows_metadata[
        (windows_metadata['end_index'] - windows_metadata['start_index'] + 1) == expected_rows
        ]

    return complete_windows_metadata


def create_3d_matrix(data, groupby, features, labels_window_agg, **kwargs):
    """Creates the 3d matrices with data and labels.

    .. warning: Ensure to do any filling before!

    Parameters
    ----------
    data:
    groupby:
    features:
    labels_window_agg:

    Returns
    -------
    """
    # Get sizes
    n_samples = len(data.groupby(groupby))
    n_steps = int(len(data) / len(data.groupby(groupby)))
    n_features = len(features)

    # Reshape
    X = data[features].values \
        .reshape(n_samples, n_steps, n_features)

    # Create DataFrame with labels
    df_y = extracted_data_df \
        .groupby(groupby) \
        .agg(labels_window_agg)

    # Create numpy matrix
    y = df_y.fillna(False) \
        .astype(int).values

    # Return
    return X, y, data, df_y




if __name__ == '__main__':

    # Libraries
    import json
    from dotmap import DotMap
    from pathlib import Path
    from tqdm import tqdm
    from functools import partial
    from prep import reshape

    tqdm.pandas()

    # ------------------------------
    # Helper methods
    # ------------------------------
    # ..
    def prop(x, pct=0.5):
        return (np.sum(x) / len(x)) > pct

    prop50 = partial(prop, pct=0.5)
    prop90 = partial(prop, pct=0.75)


    def fcast(x, at=0):
        return x.values[-at]

    fcast1 = partial(fcast, at=1)
    fcast2 = partial(fcast, at=2)
    fcast3 = partial(fcast, at=3)
    fcast4 = partial(fcast, at=4)

    # -----------------------------
    # Config
    # -----------------------------
    RUN_MANUAL = True
    RUN_DATA = True

    if RUN_MANUAL:

        # ---------------------------
        # Configuration
        # ---------------------------
        CONFIG = DotMap({
            'window': {
              'look_back': 2,
              'look_ahead': 2,
              'step': 1
            },
            'matrix': {
                'groupby': 'window_id',
                'metadata': ['say_id', 'stay_time'],
                'features': ['feature_1', 'feature_2'],
                'labels': ['label_1', 'label_2'],
                'labels_window_agg': {
                    'label_1': ['max'],
                    'label_2': [('prop50', lambda x: prop(x, pct=0.5)),
                                ('prop90', lambda x: prop(x, pct=0.9)),
                                ('fcast1', lambda x: fcast1(x, at=1)),
                                ('fcast2', lambda x: fcast2(x, at=2)),
                                ('fcast3', lambda x: fcast3(x, at=3)),
                                ('fcast4', lambda x: fcast4(x, at=4))]
                }
            }
        })

        # ---------------------------
        # Create test samples
        # ---------------------------
        # Example DataFrame
        data = {
            'stay_id': [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            'stay_time': [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8],
            'feature_1': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160],
            'feature_2': [15, 25, 35, 45, 55, 65, 75, 85, 95, 100, 105, 110, 115, 120, 125, 130],
            'label_1': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            'label_2': [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        }

        # Create DataFrame
        df = pd.DataFrame(data)

        # ---------------------------
        # Compute windows
        # ---------------------------
        # Create window metadata
        window_metadata_df = generate_window_metadata(df, **CONFIG.window)
        # Extract windows from data
        extracted_data_df = extract_window_data(df, window_metadata_df)
        # Create DataFrames and matrices
        X, y, df_x, df_y = create_3d_matrix(extracted_data_df, **CONFIG.matrix)

        # Show
        print(df_x)
        print(df_y)
        print(X.shape)
        print(y.shape)



    if RUN_DATA:

        metadata = ['stay_id', 'stay_time']
        features = [
            # statics
            'age_static', 'female_static',
            'weight_static', 'height_static',
            # vital signs (temporal)
            'map', 'hr', 'temp', 'sbp',
            'dbp', 'o2sat', 'resp',
            # bio-markers (statics)
            'glu', 'lact', 'alp',  'hct', 'hgb', 'plt',
            'etco2', 'po2', 'tco2', 'pco2', 'cai',
            'ca', 'crea', 'bildir', 'tnt',
            'bili', 'tri', 'na',  'alb', 'alt', 'ck', 'ckmb',
            'crp', 'be', 'bicar', 'ph', 'cl', 'mg', 'phos',
            'k', 'ast', 'bun', 'ptt', 'fgn',
            'bnd',  'esr', 'hbco', 'inrpt','methb', 'pt',
            'lymph', 'mch', 'mchc', 'mcv', 'neut',
            'rbc', 'rdw', 'wbc', 'eos', 'basos',
            # context
            'urine',  'ventialtion',
            'vasopressors', 'vent_dur',
            'samp', 'abx',
        ]
        labels = [
            'si_on_moor2023', 'si_on_moor2023_48_24',
            'bac_on_bahp2024', 'bac_on_bahp2024_0_24',
            'sep_on_moor2023', 'sep_on_moor2023_0_24'
        ]


        # ---------------------------
        # Configuration
        # ---------------------------
        CONFIG = DotMap({
            'window': {
                'look_back': 6,
                'look_ahead': 4,
                'step': 1
            },
            'matrix': {
                'groupby': 'window_id',
                'metadata': ['stay_id', 'stay_time', 'abx', 'samp', 'bsofa', 'dsofa'],
                'features': features,
                'labels': labels,
                'labels_window_agg': {
                    'si_on_moor2023': ['max'],
                    'sep_on_moor2023': ['max'],
                    'sep_on_moor2023_0_24': [
                        ('prop50', lambda x: prop(x, pct=0.5)),
                        ('prop90', lambda x: prop(x, pct=0.9)),
                        ('fcast1', lambda x: fcast1(x, at=1)),
                        ('fcast2', lambda x: fcast2(x, at=2)),
                        ('fcast3', lambda x: fcast3(x, at=3)),
                        ('fcast4', lambda x: fcast4(x, at=4))]
                }
            }
        })


        # ------------------------------
        # Definition
        # ------------------------------
        # Data to load
        DATAPATH = Path('C:\\Users\\kelda\\Desktop\\datasets\\ricu\\mac')
        SRC = 'mimic'
        DB = '%s_0.5.6_lbl.parquet' % SRC
        FILEPATH = DATAPATH / DB

        # ------------------------------
        # Main
        # ------------------------------
        # Load data
        data = pd.read_parquet(FILEPATH)
        data.stay_time = data.stay_time.astype(int)
        print(data.columns.tolist())

        aux = data #.head(10000)
        #aux = data[data.stay_id.isin([200003])]
        #aux = data.head(50000)
        #aux = aux[(aux.stay_time >= 0) & (aux.stay_time <= 72)]

        # ---------------------------
        # Compute windows
        # ---------------------------
        # Create window metadata
        window_metadata_df = generate_window_metadata(aux, **CONFIG.window)
        # Extract windows from data
        extracted_data_df = extract_window_data(aux, window_metadata_df)
        # Create DataFrames and matrices
        X, y, df_x, df_y = create_3d_matrix(extracted_data_df, **CONFIG.matrix)

        # Show
        print(df_x)
        print(df_y)
        print(X.shape)
        print(y.shape)


        """
        # Compute window length
        first_row = window_metadata_df.iloc[0]
        n_steps = first_row.end_index - first_row.start_index
        
        """

        # Add more info so that it is saved in config
        CONFIG.data.columns = extracted_data_df.columns.tolist()
        CONFIG.data.index = extracted_data_df.index.tolist()
        CONFIG.dfy.columns = df_y.columns.tolist()
        CONFIG.dfy.index = df_y.index.tolist()
        CONFIG.dataset.DATAPATH = Path('C:\\Users\\kelda\\Desktop\\datasets\\ricu\\mac')
        CONFIG.dataset.SRC = 'mimic'
        CONFIG.dataset.DB = '%s_0.5.6_lbl.parquet' % SRC
        CONFIG.dataset.FILEPATH = DATAPATH / DB

        # -------------------------------------------
        # Save all
        # -------------------------------------------
        def filter_serializable(locals_dict):
            serializable_dict = {}
            for key, value in locals_dict.items():
                try:
                    json.dumps(value)  # Test if variable is serializable
                    serializable_dict[key] = value
                except (TypeError, ValueError):
                    pass  # Skip non-serializable objects
            return serializable_dict

        # Libraries
        from datetime import datetime

        # Define variables
        TIME = datetime.now().strftime("%Y%m%d-%H%M%S")
        PATH = Path('data/windows/%s' % TIME)

        # Create path if it does not exist
        PATH.mkdir(parents=True, exist_ok=True)

        # .. note: Hack to be able to save the CONFIG because the
        #          partial functions are not json serializable.
        #          It can probably be done in a better way.

        # Save config
        config_dict = CONFIG.toDict()
        del config_dict['matrix']['labels_window_agg']
        with open(PATH / 'config.json', "w") as f:
            json.dump(filter_serializable(config_dict), f, indent=4)

        # Save locals
        #with open(PATH / 'locals.json', "w") as f:
        #    json.dump(filter_serializable(locals()), f, indent=4)

        # Save windows metadata
        window_metadata_df.to_csv(PATH / 'window_metadata.csv')
        df_y.to_csv(PATH / 'df_y.csv')

        # Save numpy matrices
        np.savez(PATH / 'matrices', X=X, y=y)

        #         X_train=X_train, y_train=y_train,
        #         X_val=X_val, y_val=y_val,
        #         X_test=X_test, y_test=y_test)

        # .. warning:: Note that the matrices have the whole data from the
        #              very beginning of the look back to the end of the
        #              look ahead. Hence, further filtering needs to be done
        #              when including the data into the algorithm.
        #
        #              X_input = m[:, :look_back+1, :]
        #              X_future = m[:, look_back+1:, :
        #              y