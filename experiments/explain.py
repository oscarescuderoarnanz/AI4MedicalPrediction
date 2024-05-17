# Libraries
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from keras.models import load_model
from utils.display import plot_history
from utils.display import plot_confusion_matrix
from utils.metrics import custom_metrics_binary_clf

PATH = Path('outputs')

# -------------------
# Load data
# -------------------
# Load matrices
matrices = np.load(PATH / 'data' / 'matrices.npz')
# Select test dataset
X_test, y_test = matrices['X_test'], matrices['y_test']

print(X_test)

# -------------------
# Compute metrics
# -------------------
# Define specific paths
paths = [
    PATH / '20240517-134938/lstm/model.h5'
]

paths = list(PATH.glob(pattern='**/*.h5'))

skip_if_exists = False

# Loop
for p in paths:
    # Skip
    if skip_if_exists:
        if (p.parent / 'metrics.json').exists():
            continue

    # Load model
    model = load_model(Path(p))
    # Show model.
    print("\n\nComputing... %s" % p)
    print(model.summary())

    # Compute custom/standard metrics
    r1 = custom_metrics_binary_clf(model, X_test, y_test)
    # Save
    pd.Series(r1).to_json(Path(p).parent / 'metrics.json',
        indent=4)


    # ---------------
    # Display history
    #----------------
    # Show history
    history = pd.read_csv(p.parent / 'history.csv')
    history.epoch = history.epoch.astype(int)
    f1, axes1 = plot_history(history)

    # ---------------
    # Display metrics
    # ---------------
    # Load data
    with open(p.parent / 'metrics.json', 'r') as f:
        metrics = json.load(f)

    # Display confusion matrix
    cm = metrics['confusion_matrix']
    f2, axes2 = plot_confusion_matrix(cm)

    # Save
    (p.parent / 'figures').mkdir(exist_ok=True)
    f1.savefig(fname=p.parent / 'figures' / 'history.png')
    f2.savefig(fname=p.parent / 'figures' / 'cm.png')

    plt.show()