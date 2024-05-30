# -----------------------
# Display utils
# -----------------------

def plot_history(data):
    """Display values for each epoch

    .. note: It does not return a figure so it is
             necessary to find a workaround to save
             the image.

             axes = data.plot(x='epoch', subplots=subplots,
                layout=(rows, cols), figsize=(15, 4),
                colormap='viridis')

    Parameters
    ----------

    Returns
    -------
    """

    def get_metric_names(names):
        """Find metric names

        .. note: not working..
            cols = names[names.str.startswith('val_')]
            cols = cols.str.removeprefix('val_').tolist()
        """
        return names[~names.str.startswith('val')].tolist()[1:]


    # Libraries
    import matplotlib.pyplot as plt

    # Get the names of the metrics
    names = get_metric_names(data.columns)

    # Display
    N, cols = len(names), 5
    rows = (N // cols) + int((N % cols) > 0)
    subplots = [(n, 'val_%s' % n) for n in names]

    # Display
    f, axes = plt.subplots(rows, cols, figsize=(15, 4))
    axes = axes.flatten()
    for i, (c1,c2) in enumerate(subplots):
        data.plot(x='epoch', y=[c1,c2], ax=axes[i])

    # Configure
    for ax in axes:
        ax.locator_params(integer=True)
    plt.tight_layout()

    # Return
    return f, axes


def plot_confusion_matrix(cm, labels=None, width=3, height=2):
    """Display confusion matrix

    Parameters
    ----------

    Returns
    -------
    """
    # Libraries
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    # Dsplay
    cm = np.array(cm)
    f = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    f.plot()
    f.figure_.set_figwidth(width)
    f.figure_.set_figheight(height)

    # return
    return f.figure_, f.ax_

def plot_patient_stay(m,
                      xticklabels=None,
                      yticklabels=None,
                      colormaps=None):
    """

    .. note: Remove pandas dependency.

    Parameters
    ----------
    m: np.ndarray
        The data to display. Note that the matrix will be transposed
        before displaying. 
    xticklabels: array
    yticklabels: array
    colormaps: array

    Returns
    -------
    """
    # Libraries
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Define size
    T, F = m.shape

    # Create random matrix
    #m = np.transpose(np.random.rand(T, F))
    m = np.round(m.T, decimals=1)

    # Create parameters if not defined
    if xticklabels is None:
        xticklabels = ['%s' % i for i in range(m.shape[1])]
    if yticklabels is None:
        yticklabels = ['f%s' % i for i in range(m.shape[0])]
    if colormaps is None:
        colormaps = ['Reds'] * len(yticklabels)

    # Create figure
    f, axs = plt.subplots(m.shape[0], 1,
        figsize=(T*0.5, F*0.5), sharex=True)

    # Display
    for r in range(m.shape[0]):
        sns.heatmap([m[r,:]],
            yticklabels=[yticklabels[r]],
            xticklabels=xticklabels, annot=True, fmt='.2f',
            ax=axs[r], cmap=colormaps[r],
            linewidth=.5, annot_kws={"fontsize": 6}, cbar=False,
            square=True)
        axs[r].set_yticklabels([yticklabels[r]], rotation=0),
        axs[r].yaxis.grid(True, linestyle='dashed', color='gray')
        axs[r].set_axisbelow(True)

    """
    
    import pandas as pd
    
    # Create dataframe
    df = pd.DataFrame(m, index=yticklabels, columns=xticklabels)
    
    counter = 0
    for index, row in df.iterrows():
        sns.heatmap(np.array([row.values]),
            yticklabels=[yticklabels[counter]],
            xticklabels=xticklabels, annot=True, fmt='.2f',
            ax=axs[counter], cmap=colormaps[counter],
            linewidth=.5, annot_kws={"fontsize": 6}, cbar=False,
            square=True)
        axs[counter].set_yticklabels([yticklabels[counter]], rotation=0),
        counter += 1

        # Adjust spaces
        plt.subplots_adjust(left=0.05, right=1,
                            bottom=0.1, top=1,
                            wspace=0, hspace=0)
    """

    # Adjust spaces
    plt.subplots_adjust(left=0.05, right=1,
                        bottom=0.1, top=1,
                        wspace=0, hspace=0)

    # Return
    return f, axs








if __name__ == '__main__':

    # Libraries
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt

    # ------------------------------
    # Example display patient stay
    # ------------------------------
    # Define size
    rows, cols = 7, 10

    # Create random matrix
    m = np.transpose(np.random.rand(rows, cols))

    # Configure
    features = list('abcdfefhijklmnopqrst'[:rows])
    colormaps = ['Reds']*5 + ['Blues']*(rows-5)

    # Display
    plot_patient_stay(m,
                      xticklabels=None,
                      yticklabels=features,
                      colormaps=colormaps)
    plt.show()