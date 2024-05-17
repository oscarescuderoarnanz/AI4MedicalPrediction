# Libraries
import tensorflow as tf

from keras import regularizers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import LSTM
from keras.layers import BatchNormalization
from keras.models import load_model

from numpy.random import seed

seed(30)


"""
# .. note: Included here just for reference to remember them.
#          See: https://keras.io/api/callbacks/
#
CALLBACKS = [
    tf.keras.callbacks.CSVLogger(
        filename='filename.csv', separator=',', append=False),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patient=5, restore_best_weights=True)
]
"""

METRICS = [
    tf.keras.metrics.AUC(from_logits=True, name='aucfl'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.TruePositives(thresholds=None, name='tp', dtype=None),
    tf.keras.metrics.TrueNegatives(thresholds=None, name='tn', dtype=None),
    tf.keras.metrics.FalsePositives(thresholds=None, name='fp', dtype=None),
    tf.keras.metrics.FalseNegatives(thresholds=None, name='fn', dtype=None)
]


# -----------------------------------------------------
# Helper functions
# -----------------------------------------------------
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """Create the Transformer Encoder

    Parameters
    ----------

    Results
    -------
    """

    # Libraries
    from keras import Input
    from keras.models import Model
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Activation
    from keras.layers import GlobalAveragePooling1D
    from keras.layers import MultiHeadAttention
    from keras.layers import LayerNormalization
    from keras.layers import Conv1D
    from keras.layers import LSTM
    from keras.layers import BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    from keras import regularizers

    # Attention and Normalization
    x = MultiHeadAttention(
        key_dim=head_size,
        num_heads=num_heads,
        dropout=dropout
    )(inputs, inputs)
    x = Dropout(dropout)(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = Dropout(dropout)(x)
    x = Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    # Return
    return x + res


# -----------------------------------------------------
# Models
# -----------------------------------------------------
def binary_lstm(input_shape, metrics=METRICS):
    """Create binary classification model."""
    # Define the LSTM model
    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0., input_shape=input_shape))
    model.add(LSTM(100, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))

    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=metrics
    )

    # Return model
    return model

def binary_tfmr(input_shape,
                head_size,
                num_heads,
                ff_dim,
                num_transformer_blocks,
                mlp_units,
                dropout=0,
                mlp_dropout=0,
                metrics=METRICS):
    """Build tansformer model.

    source: https://keras.io/examples/timeseries/timeseries_classification_transformer/"""
    """"""
    # Libraries
    from keras import Input
    from keras.models import Model
    from keras.layers import Dense
    from keras.layers import Dropout
    from keras.layers import Activation
    from keras.layers import GlobalAveragePooling1D
    from keras.layers import MultiHeadAttention
    from keras.layers import LayerNormalization
    from keras.layers import Conv1D
    from keras.layers import LSTM
    from keras.layers import BatchNormalization
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping
    from keras import regularizers

    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size,
                num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    model = Model(inputs, outputs)

    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=metrics
    )

    # Return model
    return model



def multilabel_lstm(input_shape, n_outputs=1):
    """Create multi-label.
    """
    # Create model
    model = Sequential()
    model.add(tf.keras.layers.Masking(mask_value=0., input_shape=input_shape))
    model.add(LSTM(100, input_shape=input_shape, return_sequencies=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(LSTM(100))
    model.add(Dense(n_outputs, activation='sigmoid'))

    # Compile
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=[tf.keras.metrics.AUC(from_logits=True, name='auc')])

    # Return
    return model






