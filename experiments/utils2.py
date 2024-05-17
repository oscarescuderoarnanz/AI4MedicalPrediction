# Libraries
from numpy.random import seed
import tensorflow as tf

seed(30)

# -----------------------------------------------------------
# Definitions
# -----------------------------------------------------------
FEATURES = []

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


def reshape(data, features, labels, sort=False):
    """

    .. note: Sort features and labels.Thus, if the same values are
             passed the results would be the same independent of the
             order.It would be possible to include a flag parameters
             to decide whether to sort or not.

    :param data:
    :param features:
    :param labels:
    :param sort:
    :return:
    """
    # Get sizes
    n_samples = len(data.groupby('id'))
    n_steps = int(len(data) / len(data.groupby(['id'])))
    n_features = len(features)

    # Sort variables
    f_list = sorted(features) if sort else features
    l_list = sorted(labels) if sort else labels

    # Reshape
    X_data = data[f_list].fillna(0).values \
        .reshape(n_samples, n_steps, n_features)
    y_data = data.groupby(['id'])[l_list] \
        .max().astype(int).values

    # Return
    return X_data, y_data


def custom_metrics(model, X, y):
    """"""
    # Libraries
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import average_precision_score

    # Running saved model against the data
    predictions = model.predict(X)

    # Metrics
    return {
        'roc': roc_auc_score(y, predictions),
        'aps': average_precision_score(y, predictions)
    }



# -----------------------------------------------------------
#                        Shortcuts
# -----------------------------------------------------------
# .. note: There is an option. of passing the variables using
#          **locals(). However, this will also pass variables
#          which have been created within the function.

def sep(*args, **kwargs):
    """"""
    return sepsis(*args, **kwargs)

def bac(args, **kwargs):
    """"""
    return bacteremia(*args, **kwargs)

def bsi(args, **kwargs):
    """"""
    return bloodstream_infection(*args, **kwargs)



def sepsis(a=1, b=2, c=3):
    """"""
    print(a, b, c)
    pass

def bacteremia():
    """"""
    pass

def bloodstream_infection():
    """"""
    pass