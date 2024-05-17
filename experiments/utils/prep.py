# Libraries

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


# --------------------------------------------------------------
#                     Features / Labels
# --------------------------------------------------------------
# The methods included in this section should be related with
# the creation of hand-crafted features (e.g. PaO2/fiO2) and/or
# any complex labels (e.g. bacteremia, sepsis, ...)

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

def pf_ratio(pao2, fio2):
    """Computes the fraction of inspired oxygen.

    Paramters
    ---------
    pao2: np.array
        The partial pressure of oxygen in arterial blood (what unit?)
    fio2: np.array
        The fraction of inspiratory oxygen concentration (what unit?

    Returns
    paO2
    """
    pass



# --------------------------------------------------------------
# Other methods to setup the experiments
# --------------------------------------------------------------
"""
def scale_data(a, b):
    # Libraries
    from sklearn.preprocessing import MinMaxScaler

    # Create instance
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Cre
"""


def reshape(data, by, features, labels, sort=False):
    """Reshape data for ... ????

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
    n_samples = len(data.groupby(by))
    n_steps = int(len(data) / len(data.groupby(by)))
    n_steps = 7
    n_features = len(features)

    # Sort variables
    f_list = sorted(features) if sort else features
    l_list = sorted(labels) if sort else labels

    # Reshape
    X_data = data[f_list].fillna(0).values \
        .reshape(n_samples, n_steps, n_features)
    y_data = data.groupby(by)[l_list] \
        .max().astype(int).values

    # Return
    return X_data, y_data