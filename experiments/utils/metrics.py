# Libraries
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_score
from sklearn.metrics import brier_score_loss
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer



# ----------------------------------------------
# Metrics
# ----------------------------------------------
def metric_oscar(b):
    """

    >>> metric_oscar(3) -> 6

    :return:
    """
    return b+3


# ----------------------------------------------
# Combined
# ----------------------------------------------
# ----------------------------------------------------
# Helper methods
# ----------------------------------------------------
def custom_metrics_binary_clf(model, X, y):
    """Compute custom metrics for binary classification problems.

    Parameters
    ---------
    model:
    X: np.array or pd.DataFrame
        The data with the features
    y: np.array or pd.DataFrame
        The data with the labels

    Returns
    -------
    """
    # Run predictions
    y_prob = model.predict(X)
    y_pred = y_prob > 0.5

    # Compute opt thresold
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    J = tpr - fpr
    ix = np.argmax(J)
    best_thresh = thresholds[ix]

    # Class using bess threshold
    y_thrs = y_prob > best_thresh

    # Other metrics
    cr = classification_report(y, y_thrs, output_dict=True)
    cm = confusion_matrix(y, y_thrs)
    tn, fp, fn, tp = cm.ravel()

    # Return
    return {
        'roc': roc_auc_score(y, y_prob),
        'aps': average_precision_score(y, y_prob),
        'best_threshold': best_thresh,
        'sens': recall_score(y, y_thrs, pos_label=1),
        'spec': recall_score(y, y_thrs, pos_label=0),
        'ppv': precision_score(y, y_thrs, pos_label=1),
        'npv': precision_score(y, y_thrs, pos_label=0),
        'brier': brier_score_loss(y, y_prob),
        'classification_report': cr,
        'confusion_matrix': cm,
    }





if __name__ == '__main__':

    # Libraries

    # Test/Va.
    metric_oscar()