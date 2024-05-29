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


# ----------------------------------
# Helper methods
# ----------------------------------
def propagate(x, bck=0, fwd=0, force=False):
    """Propagates values (uses fillna)

    .. note: Propagate only fills NaN!

    Parameters
    ----------
    x: pd.Series
        The vector to propagate
    bck: int
        Propagate backwards (>0)
    fwd: int
        Propagate forwards (>0)
    """
    name = '%s_%s_%s' % (x.name, bck, fwd)
    c = x.copy().rename(name)
    if fwd > 0:
        c = c.ffill(limit=fwd)
    if bck > 0:
        c = c.bfill(limit=bck)
    #if force:
    #print("Forcing!")
    #c = x.copy() == True
    #true_index = x[x.abx == True].index[0]
    #start_index = max(0, true_index - Nts_pre)
    #end_index = min(len(df), true_index + Nts_post + 1)
    #df.loc[start_index:end_index, 'SI'] = 1


    return c


# --------------------------------------------------------------
#
# --------------------------------------------------------------
# The methods included in this section should be related with
# the creation of hand-crafted features (e.g. PaO2/fiO2) and/or
# any complex labels (e.g. bacteremia, sepsis, ...)

def si_basic(x, prop=None):
    """Suspected infection (just abxs)"""
    # Check required columns
    if not 'abx' in x:
        print("Raise warning or error!")
    # Compute
    c = x.abx.copy().rename('si')
    if prop is not None:
        c = propagate(c, prop[0], prop[1])
    return c

def si_persson_2021(x):
    """
    Suspected infections are instances in which antibiotics have
    been prescribed and when body fluid cultures were present in
    the electronic health record if a culture is ordered within
    24 hours after antibiotics, or antibiotics had been prescribed
    less than 72 hours after a culture order.

    .. note: see sepsis-relevant antibiotics

    Parameters
    ----------

    Returns
    -------
    """
    if not 'abx' in x or not 'samp' in x:
        print("[Error] Raise!")
    # Compute
    #abxs = propagate(x.abx.copy(), bck=bck, fwd=fwd)
    #samp = propagate(x.samp.copy(), bck=bck, fwd=fwd)
    #return pd.Series(abxs | samp, name='si_lin')
    pass


def si_lin_2021(x, bck=0, fwd=0):
    """Suspected infection (antibiotics or sample).

    Suspected infections are instances in which either antimicrobial
    prescription or specimen collection is found in electronic health
    records.

    .. warning:: Weird behaviour
             True | None = True
             None | True = False

    .. note: Since they do not often happen exactly at the same hour,
             backward and forward windows can be used to propagate
             the values of <abxs> and <samp>.

    Parameters
    ----------
    """
    if not 'abx' in x or not 'samp' in x:
        print("[Error] Raise!")
    # Compute
    abxs = propagate(x.abx, bck=bck, fwd=fwd)
    samp = propagate(x.samp, bck=bck, fwd=fwd)
    sinf = pd.Series(abxs | samp, name='si_lin_%s_%s' % (bck, fwd))
    #print(pd.concat([abxs, samp, sinf], axis=1))
    #df.where(df.notna().cumprod(axis=1).eq(1))
    return pd.Series(abxs | samp, name='si_lin_%s_%s' % (bck, fwd))


def si_valik_2023(x):
    """
    Suspected infection are instances in which either two antimicrobial
    doses were administered or specimen collection was found in EHR or
    an increase of SOFA >= 2 compared to baseline.

    What is baseline?
    It refers to the SOFA score typically at the time of ICU admission
    or the point at which they were first evaluated.

    Parameters
    ----------
    """
    pass


def suspected_infection(x, strategy='basic', **kwargs):
    """Computes whether there is suspected infection.

    .. note: Whether abxs were applied.
    .. note: Whether cultures were requested.

    Possible elements to define suspected infection:
      - antibiotics have been prescribed &
      - blood fluid cultures applied (vasopressors?)
      - culture order within 24h after antibiotics
      -


    Parameters
    ----------
    x: pd.DataFrame
        The DataFrame with information to compute the suspected infection.
        It must include the following
    strategy: str
        The strategy to use to compute the suspected infection. The
        options are:
            - basic:
            - si_persson_2021
            - si_lin_2021
            - si_valik_2023
    prop: tuple
        The number of time steps to propagate the label.
        In order to avoid any propagation, the parameter
        must be None.

    Returns
    -------
    """
    if strategy == 'basic':
        return si_basic(x, **kwargs)
    elif strategy == 'lin_2021':
        return si_lin_2021(x, **kwargs)
    elif strategy == 'persson_2021':
        return si_persson_2021(x, **kwargs)
    elif strategy == 'valik_2023':
        return si_valik_2023(x, **kwargs)
    else:
        print("ERROR!")



def bacteremia():
    """Computes bacteremia onset.

    It requires information about the microbiology results. It checks
    that the cultures have grown a pathogen which is not one of the
    considered common contaminants.

    .. note: Positive microbiology culture.
    .. note: Exclude common contaminants.

    Parameters
    ----------

    Returns
    -------
    """
    pass



def bloodstream_infection():
    """Computes blood stream infection onset.

    It is defined as the growth of a clinically significant pathogen
    in at least one blood culture. Potential contaminates can be
    defined according to the following guidelines:
      - CDC/NHSN
      - CLSI

    .. note: bacteremia + suspected infection

    Parameters
    ----------

    Returns
    -------
    """
    pass


def sep_moor_2023(x, delta_sofa=2,
                     bck_abx=0, fwd_abx=24,
                     bck_smp=0, fwd_smp=72,
                     bck_si=48, fwd_si=24):
    """

    The Sepsis-3 criterion defined sepsis as co-occurrence of suspected
    infection and a SOFA increase of two or more points. We followed the
    approach of the original authors as closely as possible. Suspected
    infection was defined as co-occurrence of antibiotic treatment and
    body fluid sampling. If antibiotic treatment occurred first, it needed
    to be followed by fluid sampling within 24 hours. Alternatively, if
    fluid sampling occurred first, it needed to be followed by antibiotic
    treatment within 72 hours in order for it to be classified as suspected
    infection. The earlier of the two times is taken as the suspected
    infection (SI) time. After this, the SI window is defined to begin 48
    hours before the SI time and end 24 hours after the SI time.

    What is fluid sampling?
      - Just any fluid, for example for a full blood count.
      - A fluid sample send to the microbiology lab (culture).

    Is the increase related to the baseline?

    .. note: We could do it by steps.
         - identify which is first abx or smp
         - propagate the corresponding and see if overlaps
         - define SI extended
         - define sepsis

    Parameters
    ----------

    Returns
    -------
    """

    def f_empty_onset(s):
        return pd.Series(None, index=s.index, name='si_onset_moor')

    def f_baseline_sofa(s):
        if s.isna().all():
            return pd.Series(None, index=s.index, name='bsofa')
        return pd.Series(s - s[s.first_valid_index()], name='bsofa')

    def f_delta_sofa(s, **kwargs):
        return pd.Series(s.diff(**kwargs), name='dsofa')

    def f_empty_dataframe(s):
        print(s)
        s1 = f_empty_onset(s)
        s2 = propagate(s1, bck=bck_si, fwd=fwd_si)
        s3 = f_delta_sofa(s)
        s4 = f_baseline_sofa(s)
        return pd.concat([s1, s2, s4, s3], axis=1)

    if not 'abx' in x or not 'samp' in x:
        print("[Error] Raise!")

    # There are no antibiotics nor samples.
    if x.abx.isna().all() | x.samp.isna().all():
        return f_empty_dataframe(x.sofa)

    # Check whether antibiotics and samples overlap.
    abxs = propagate(x.abx, bck=bck_abx, fwd=fwd_abx)
    samp = propagate(x.samp, bck=bck_smp, fwd=fwd_smp)
    overlap = abxs & samp

    if not overlap.any():
        return f_empty_dataframe(x.sofa)

    # Create SI onset (first of abx or samp)
    si_onset = f_empty_onset(x.sofa)
    idx1 = np.argmax(x.abx == True)
    idx2 = np.argmax(x.samp == True)
    si_onset.iloc[min(idx1, idx2)] = True

    # Propagate SI onset
    si_moor = propagate(si_onset, bck=bck_si, fwd=fwd_si)

    # Compute baseline sofa increment
    bsofa = f_baseline_sofa(x.sofa)
    dsofa = f_delta_sofa(x.sofa)

    sep_onset = pd.Series(None, index=x.index, name='sep_onset')
    sep_onset[(bsofa >= delta_sofa) & si_moor] = 1

    # Return
    return pd.concat([si_onset, si_moor, sep_onset, dsofa, bsofa], axis=1)


def sepsis(x, delta_sofa=2, prop=None):
    """Computes sepsis onset.

    .. note: Sepsis onset defined as increase in sofa > 2
             and suspicion of infection.

    SEPSIS-1: SIRS
    SEPSIS-2: SIRS + SI
    SEPSIS-3: SOFA + SI

    Possible elements to define sepsis:
       - ICD-10: codes
       - ICD-9: 995.91 (sepsis) | 995.92 (severe sepsis) | 785.52 (septic shock)
       - delta_sofa > 2 and suspected infection
       -

    SI: Suspected Infection

    Parameters
    ----------
    x: pd.DataFrame
        The DataFrame with the information.
    delta_sofa: number
        The increase in consecutive sofas.
    prop: tuple
        The propagation

    Returns
    -------
    """
    # Libraries
    import pandas as pd
    import numpy as np

    # Basic checks
    if not 'sofa' in x or not 'si':
        print("Raise warning!")

    # Compute delta sofa.
    dsofa = pd.Series(x.sofa.diff(), name='dsofa')
    # Compute sepsis onset
    onset = pd.Series(np.nan, index=x.index, name='sep_onset')
    onset[(dsofa >= delta_sofa) & x.si] = 1
    # Compute propagation
    if prop is not None:
        extnd = propagate(onset, prop[0], prop[1])
        return pd.concat([dsofa, onset, extnd], axis=1)
    # Return
    return pd.concat([dsofa, onset], axis=1)




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
    return X_data, y_data#



def get_SI(df, Nts_pre, Nts_post):
    df['SI'] = 0
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





if __name__ == '__main__':

    # Libraries
    import numpy as np
    import pandas as pd

    # -------------------
    # Create test samples
    # -------------------
    n, t, f = None, True, False

    p1 = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [n, n, t, n, n, n, n, n, n, n],
        [n, n, t, n, n, n, n, n, n, n],
        [1, 2, 3, 4, 5, 4, 3, 2, 1, 1]
    ]).T

    p2 = np.array([
        [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [n, n, t, n, n, n, n, n, n, n],
        [n, n, n, n, t, n, n, n, n, n],
        [1, 1, 5, 6, 7, 1, 1, 4, 1, 1]
    ]).T

    p3 = np.array([
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [n, n, n, n, n, t, n, n, n, n],
        [n, n, t, n, n, n, n, n, n, n],
        [1, 1, 5, 1, 9, 1, 2, 1, 1, 1]
    ]).T

    p4 = np.array([
        [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [n, t, n, n, n, n, n, n, n, n],
        [n, n, n, n, n, n, n, n, t, n],
        [1, 1, 5, 1, 9, 1, 2, 1, 1, 1]
    ]).T

    p5 = np.array([
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [t, n, n, n, n, n, n, n, t, n],
        [n, t, n, n, n, n, n, t, n, n],
        [1, 1, 5, 1, 9, 1, 2, 1, 1, 1]
    ]).T


    p6 = np.array([
        [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        [n, n, n, n, n, n, n, n, n, n],
        [n, n, n, n, n, n, n, n, n, n],
        [1, 1, 5, 1, 9, 1, 2, 1, 1, 1]
    ]).T

    # Combine samples
    m = np.concatenate((p1, p2, p3, p4, p5, p6))

    # .. note: Be careful when using convert_dtypes() as the
    #          comparison between the pd.NA is different to
    #          the comparison with None

    # Create DataFrame
    dtypes = {
        'stay_id': int,
        'stay_time': int,
        'abx': bool,
        'samp': bool,
        'sofa': float
    }

    # Create DataFrames
    df1 = pd.DataFrame(p1, columns=dtypes.keys())
    df2 = pd.DataFrame(m, columns=dtypes.keys())

    #aux = aux.astype(dtypes)
    #aux.abx = aux.abx.astype(int)


    # -------------------
    # Propagate
    # -------------------
    r0 = propagate(df1.abx, bck=1, fwd=1)


    # ------------------------------------------------
    # Compute suspected infection
    # ------------------------------------------------
    # Compute and propagate manually
    r0 = suspected_infection(df1)
    r1 = propagate(r0, bck=1, fwd=2)
    r2 = propagate(r0, bck=2, fwd=3)

    # Compute with propagation
    r3 = suspected_infection(df1, prop=(2, 4))

    # Compute with group and propagation
    r4 = df2.groupby(by='stay_id').apply(
        suspected_infection, prop=(1,3),
        include_groups=False
    ).droplevel(0)

    # Compute si_lin (not working!)
    r5 = df2.groupby(by='stay_id').apply(
        si_lin_2021, bck=1, fwd=1,
        include_groups=False
    ).droplevel(0)

    # Show
    lbl = 'Suspected Infection'
    print("\n%s\n%s\n%s" % ("=" * 80, lbl, "=" * 80))
    print("\nSuspected infection (manual propagation)")
    print(pd.concat([df1, r0, r1, r2, r3], axis=1))
    print("\nSuspected infection (group propagation)")
    print(pd.concat([df2, r4, r5], axis=1))


    # -----------------------------------------------
    # Compute sepsis
    # -----------------------------------------------
    # Add suspected infection
    df1['si'] = r1

    # Compute and propagate manually
    s0 = sepsis(df1)
    s1 = propagate(s0.sep_onset, bck=1, fwd=2)
    s2 = propagate(s0.sep_onset, bck=2, fwd=3)

    # Compute with propagation
    s3 = sepsis(df1, delta_sofa=2, prop=(1, 4))


    # Compute with group and propagation
    df2['si'] = df2.groupby(by='stay_id').apply(
        suspected_infection, prop=(1,3),
        include_groups=False
    ).droplevel(0)

    s4 = df2.groupby(by='stay_id').apply(
        sepsis, prop=(1,3),
        include_groups=False
    ).droplevel(0)

    s5 = df2.groupby(by='stay_id').apply(
        sep_moor_2023, bck_abx=0, fwd_abx=2,
        bck_smp=0, fwd_smp=6, bck_si=4, fwd_si=2,
        include_groups=False
    ).droplevel(0)


    print("\n%s\nSepsis\n%s" % ("="*80, "="*80))
    print("\nSepsis (manual propagation)")
    print(pd.concat([df1, s0, s1, s2], axis=1))
    print("\nSepsis (auto propagation)")
    print(pd.concat([df1, s3], axis=1))
    print("\nSepsis (group propagation)")
    print(pd.concat([df2, s5], axis=1))

    # -----------------------------------------------
    # Compute bacteremia onset
    # -----------------------------------------------

    # -----------------------------------------------
    # Compute bloodstream infection onset
    # ---------------------------------------------


    # -----------------------------------------------
    #
    # -----------------------------------------------
    # Libraries
    import settings
    from pathlib import Path


    # Show information
    print("\nDATAPATH: %s" % settings.DATAPATH)

    # Data
    DATAPATH = Path('C:\\Users\\kelda\\Desktop\\datasets\\ricu\\win')

    # Define database
    db = 'eicu_demo_cmb_0.5.6.parquet'

    # Load data
    df = pd.read_parquet(DATAPATH / db)


    print(df.sum())

    print("samp in columns:", 'samp' in df.columns.tolist())

    # -----------------
    # Define body fluid
    # -----------------
    # Get all columns that need fluid sample
    #columns = settings.Variables() \
    #    .vars_in_categories(['hemo', 'vitals'])
    #columns = np.char.add(columns, '_raw')

    # .. note: There might be an issue computing it this
    #          way if the values of the different markers
    #          have been propagated before.

    # Compute body fluid column
    #df['body_fluid'] = df[columns].notna().any(axis=1)
    #print(df.body_fluid.value_counts())
    #print(df.shape)
    #print(df.columns.values)

    # Compute with group and propagation
    #r4 = df2.groupby(by='stay_id').apply(
    #    si_lin_2021, bck=1, fwd=1,
    #    include_groups=False
    #).droplevel(0)


    # Compute with group and propagation
    r5 = df.groupby(by='stay_id').apply(
        sep_moor_2023,
        include_groups=False
    ).droplevel(0)

    print("example!")
    aux = pd.concat([df, r5], axis=1)
    aux.to_parquet('oddio.parquet')

    import sys
    sys.exit()

    # ------------------------------------------------
    # Display to verify
    # ------------------------------------------------
    # Library
    import display
    import matplotlib.pyplot as plt

    # Create DataFrame
    df = pd.concat([df2, s4], axis=1)
    # Group by patients
    groups = df.groupby(by='stay_id', as_index=False)
    # Display
    for i, (stay_id, p) in enumerate(groups):
        # Log
        print("%s/%s. Displaying... %s" % (i, len(groups), stay_id))

        # Stop
        if i > 2:
            break

        # Format matrix
        m = p.drop(columns=['stay_id'])
        m.abx.fillna(0, inplace=True)
        m = m.convert_dtypes()  # .fillna(0)

        # Display
        f, ax = display.plot_patient_stay(
            m=m.to_numpy(dtype=np.float32),
            xticklabels=m.stay_time.astype(int).tolist(),
            yticklabels=m.columns.tolist(),
            colormaps=['Greens'] + ['coolwarm'] * 4,
            blabel=stay_id
        )

        # Show
        plt.show()


    import sys
    sys.exit()




    # ------------------------------------------------
    # Compute sepsis onset
    # ------------------------------------------------


    sep_def = {
        # Parameteres for antibiotic propagation
        'Nts_pre': 24, 'Nts_post': 24,
        # Parametere for sepsis propagation
        'N_prog_sep': 12,
        # Parameter for determine sepsis onset
        'increm_sofa': 2
    }

    df = pd.DataFrame(m, columns=['stay_id', 'sofa', 'abx'])
    df = get_SI(df, Nts_pre=24, Nts_post=24)
    df = get_sep(df, N_prog_sep=12, increm=2)
    print(df)

    # -----------------------------------------------
    # Compute bacteremia onset
    # -----------------------------------------------

    # -----------------------------------------------
    # Compute bloodstream infection onset
    # -----------------------------------------------