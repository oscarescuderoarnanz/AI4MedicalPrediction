# Libraries
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from datetime import datetime
from utils import settings
from utils import prep

#
print(settings.DATAPATH)

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------
# Select dataset
db = 'eicu_0.5.6.parquet'

# ----------------------------------------------------
# Helper methods
# ----------------------------------------------------



def load_data_scaled(df, metadata, features):
    """

    .. note: This is just a toy example.

    """
    # Libraries
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    # Copy dataFrame
    aux = df.copy(deep=True)

    # .. note: We should be considering the onset of sepsis time,
    #          and creating a sliding window. For now we are just
    #          getting the first time steps from admission.

    # Filter only one window per patient.
    aux = aux[(aux.stay_time >= 0) & (aux.stay_time <= 6)]
    # Keep those that have a full window of length n.
    vc = aux.stay_id.value_counts()
    aux = aux[aux.stay_id.isin(vc[vc == 7].index)]

    # Keep only thse with length 6



    # Missing values and create label.
    aux = aux[metadata + features]
    aux[features] = aux[features].ffill().bfill().fillna(0)
    aux.sofa = aux.sofa.ffill().bfill().fillna(0)
    aux['label'] = df.sofa > 7

    # Divide patients in three groups
    train, validate, test = \
        np.split(aux.stay_id.sample(frac=1, random_state=42),
                 [int(.6 * len(aux)), int(.8 * len(aux))])


    train = aux[aux.stay_id.isin(train)]
    validate = aux[aux.stay_id.isin(validate)]
    test = aux[aux.stay_id.isin(test)]

    # Scale
    scaler = MinMaxScaler(feature_range=(-1, 1))
    train[features] = scaler.fit_transform(train[features])
    validate[features] = scaler.transform(validate[features])
    test[features] = scaler.transform(test[features])

    # Return
    return train, validate, test


# ----------------------------------------------------
# Load data
# ----------------------------------------------------
# Read parquet file
df = pd.read_parquet(settings.DATAPATH / db)

# Add a fake label
print(df.sepshk_der.unique())
print(df.sofa.unique())
#print(df)

df['lbl_fake1'] = df.sofa.ffill().bfill() > 7

#
metadata = [
    'stay_id', 'stay_time', 'sofa'
]

features = [
    'hr_raw', 'o2sat_raw', 'temp_raw',
    'sbp_raw', 'dbp_raw', 'map_raw',
    'resp_raw', 'fio2_raw'
]

label = 'label'

# .. note: At the moment we are doing the pre-processing every time
#          we load the data because it is a demo example. However,
#          at some point the data will be already preprocessed and
#          will be loaded from file.

# Load dataset splits
train, val, test = load_data_scaled(df,
    metadata=metadata,  features=features)

# -------------------------------------------
# Train the models
# -------------------------------------------
# Libraries
from utils.models import binary_lstm
from utils import prep

# Get running time
TIME = datetime.now().strftime("%Y%m%d-%H%M%S")

# Output path
PATH = Path('./outputs')

# Get data
X_train, y_train = prep.reshape(train, ['stay_id'], features, ['label'])
X_val, y_val = prep.reshape(val, ['stay_id'], features, ['label'])
X_test, y_test = prep.reshape(test, ['stay_id'], features, ['label'])


np.savez(PATH / 'data' / 'matrices',
    X_train=X_train, y_train=y_train,
    X_val=X_val, y_val=y_val,
    X_test=X_test, y_test=y_test)


# Create models
lstm = binary_lstm(input_shape=X_train.shape[1:])

pairs = [
    ('lstm', lstm)
]

for name, model in pairs:
    # Define fullpath
    fullpath = PATH / TIME / name
    # Create folder if it does not exist
    fullpath.mkdir(parents=True, exist_ok=True)
    # Loging information
    print("Training... %s" % str(fullpath))

    # Fit the model
    history = model.fit(X_train, y_train,
        validation_data=(X_val, y_val), epochs=300,
        batch_size=256,
        callbacks=[
            tf.keras.callbacks.CSVLogger(fullpath / 'history.csv',
                separator=',', append=False),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                patience=5, restore_best_weights=True)
        ])

    # Save the model
    model.save(fullpath / 'model.h5')
    model.save(fullpath / 'model.keras')
