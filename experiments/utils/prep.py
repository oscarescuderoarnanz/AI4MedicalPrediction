# Libraries
import pandas as pd

# Warning on propagate; downcast on fill, bfill
pd.set_option('future.no_silent_downcasting', True)


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
def series_with_none(size, index=None, name=''):
    """"""
    return pd.Series([None]*size, index=index, name=name), # or None

def propagate_name(name, bck=0, fwd=0):
    """"""
    return '%s_%s_%s' % (name, bck, fwd)

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
    name = propagate_name(x.name, bck, fwd)
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

def si_moor_2023_aux(x, bck_abx, fwd_abx, bck_si, fwd_si, **kwargs):
    """Suspected infection.

    .. note:: The eICU dataset reports only a small number of body fluid
              samplings, while the HiRID dataset reports no body fluid
              samplings at all. For this reason, the original definition
              of suspected infection is hard to implement on these datasets.

    The alternative suspected infection was defined as a co-occurrence of
    multiple antibiotic administrations.
    """
    pass

def si_moor2023(x, bck_abx=0, fwd_abx=24,
                   bck_smp=0, fwd_smp=72,
                   bck_si=48, fwd_si=24):
    """Suspected infection.

    .. note:: Only detects one suspected infection (SI) event (first) during
              the whole stay! Would it be possible to define possible scenarios
              in which subsequent events happen?

    Suspected infection was defined as co-occurrence of antibiotic treatment
    and body fluid sampling. If antibiotic treatment occurred first, it needed
    to be followed by fluid sampling within 24 hours. Alternatively, if fluid
    sampling occurred first, it needed to be followed by antibiotic treatment
    within 72 hours in order for it to be classified as suspected infection.
    The earlier of the two times is taken as the suspected infection (SI) time.
    After this, the SI window is defined to begin 48 hours before the SI time
    and end 24 hours after the SI time.

    Parameters
    ----------

    Returns
    -------

    """
    name_ons = 'si_on_moor2023' # Get name automatically

    def si_empty(x):
        name_wdw = propagate_name(name_ons, bck_si, fwd_si)
        return pd.concat([
            pd.Series([None]*x.shape[0], index=x.index, name=name_ons), # or None
            pd.Series([None]*x.shape[0], index=x.index, name=name_wdw)  # or None
        ], axis=1)

    if not 'abx' in x or not 'samp' in x:
        print("[Error] Raise!")

    # There are no antibiotics nor samples.
    if x.abx.isna().all() | x.samp.isna().all():
        return si_empty(x)

    # Check whether antibiotics and samples overlap.
    abxs = propagate(x.abx, bck=bck_abx, fwd=fwd_abx)
    samp = propagate(x.samp, bck=bck_smp, fwd=fwd_smp)
    overlap = abxs & samp

    # Check overlap
    if not overlap.any():
        return si_empty(x)

    # Create SI onset (first of abx or samp)
    si_onset = pd.Series([None]*x.shape[0], index=x.index, name=name_ons)
    idx1 = np.argmax(x.abx == True)
    idx2 = np.argmax(x.samp == True)
    si_onset.values[min(idx1, idx2)] = True

    # Propagate SI onset
    si_window = propagate(si_onset, bck=bck_si, fwd=fwd_si)

    # Return
    return pd.concat([si_onset, si_window], axis=1)

    # Return with 0s
    #return pd.concat([
    #    si_onset.fillna(0).astype(int),
    #    si_window.fillna(0).astype(int)], axis=1)

    # Return with NA
    #return pd.concat([
    #   si_onset.astype('boolean'),
    #   si_window.astype('boolean')], axis=1)


def si_persson2021(x):
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


def si_lin2021(x, bck=0, fwd=0):
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


def si_valik2023(x):
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
        return si_lin2021(x, **kwargs)
    elif strategy == 'persson_2021':
        return si_persson2021(x, **kwargs)
    elif strategy == 'valik_2023':
        return si_valik2023(x, **kwargs)
    else:
        print("ERROR!")


def bac_bahp2024(x, bck_bac=2, fwd_bac=2,
                 pathogen_columns=None):
    """Computes bacteremia information.

    .. note: See second paragraph!

    The definition of bacteremia requires information about the microbiology
    results. It checks that the cultures have grown a clinically significant
    pathogen which is not one of the considered common contaminants. These are
    defined in the following guidelines: (i) CDC/NHSN, (ii) CLSI

    In order to determine the onset, at the moment, is the time the culture
    was taken; though these is not complete fair, as probably the bacteria
    was there some time before this happened.

    Parameters
    ----------

    Returns
    --------

    """
    name = 'bac_on_bahp2024'

    if pathogen_columns is None:
        bac = x.samp
    else:
        bac = x.samp & x[pathogen_columns].any(axis=1)

    # Compute
    bac_onset = pd.Series([None] * x.shape[0], index=x.index, name=name)
    if bac.any():
        bac_onset.values[np.argmax(bac == True)] = True
    bac_window = propagate(bac_onset, bck=bck_bac, fwd=fwd_bac)

    # Return
    return pd.concat([bac_onset, bac_window], axis=1)


def bacteremia():
    """Computes bacteremia onset.

    Parameters
    ----------

    Returns
    -------
    """
    pass



def bloodstream_infection():
    """Computes blood stream infection onset.

    Parameters
    ----------

    Returns
    -------
    """
    pass


def bsi_bahp2024(x, bck_abx=0, fwd_abx=24,
                    bck_smp=0, fwd_smp=72,
                    bck_si=48, fwd_si=24,
                    bck_bac=2, fwd_bac=2,
                    bck_bsi=2, fwd_bsi=2,
                    pathogen_columns=['path1', 'path2']):
    """Blood stream infection.

    .. note: Uses suspected infection from <moor 2023>.

    The blood stream infection is defined as the presence of bacteremia
    in co-ocurrence with suspicion of infection. The onset of bloodstream
    infection is set as ... ???
    """

    name = 'bsi_on_bahp2024'
    name_si_wdw = 'si_on_moor2023_%s_%s' % (bck_si, fwd_si)
    name_bac_wdw = 'bac_on_bahp2024_%s_%s' % (bck_bac, fwd_bac)


    # Compute suspected infection
    si = si_moor2023(x, bck_abx=bck_abx, fwd_abx=fwd_abx,
                        bck_smp=bck_smp, fwd_smp=fwd_smp,
                        bck_si=bck_si, fwd_si=fwd_si)

    # Compute bacteremia
    bact = bac_bahp2024(x, bck_bac=bck_bac, fwd_bac=fwd_bac,
                          pathogen_columns=pathogen_columns)

    # Compute blood stream infection
    bsi_onset = pd.Series((si[name_si_wdw] & bact[name_bac_wdw]))
    bsi_onset = bsi_onset.replace({False:None})
    bsi_onset.name = name
    bsi_window = propagate(bsi_onset, bck=bck_bsi, fwd=fwd_bsi)

    # Return
    #return pd.concat([bsi_onset, bsi_window], axis=1)

    from functools import reduce
    return reduce(lambda  left,right:
        pd.merge(left, right, left_index=True, right_index=True),
            [bsi_onset, bsi_window])



def sep_moor2023(x, delta_sofa=2,
                    bck_abx=0, fwd_abx=24,
                    bck_smp=0, fwd_smp=72,
                    bck_si=48, fwd_si=24,
                    bck_sep=0, fwd_sep=24,
                    compute_si=True):
    """

    .. note: see si_moor2023

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

    name = 'sep_on_moor2023'
    name_si_wdw = 'si_on_moor2023_%s_%s' % (bck_si, fwd_si)

    def f_baseline_sofa(s):
        if s.isna().all():
            return pd.Series(None, index=s.index, name='bsofa')
        return pd.Series(s - s[s.first_valid_index()], name='bsofa')

    def f_delta_sofa(s, **kwargs):
        return pd.Series(s.diff(**kwargs), name='dsofa')

    # Compute suspected infection
    si = si_moor2023(x, bck_abx=bck_abx, fwd_abx=fwd_abx,
                        bck_smp=bck_smp, fwd_smp=fwd_smp,
                        bck_si=bck_si, fwd_si=fwd_si)

    # Compute baseline and continuous sofa increments
    bsofa = f_baseline_sofa(x.sofa)
    dsofa = f_delta_sofa(x.sofa)

    # Define sepsis
    sep_wdw = pd.Series((bsofa >= delta_sofa) & si[name_si_wdw],
        name='sep_wdw_moor2023')

    # Define sepsis onset (first appearance) # use idxmax()
    sep_onset = pd.Series([None] * x.shape[0], index=x.index, name=name)
    if sep_wdw.any():
        sep_onset.values[np.argmax(sep_wdw == True)] = True

    # Propagate SI onset
    sep_prop = propagate(sep_onset, bck=bck_sep, fwd=fwd_sep)

    # Return
    return pd.concat([si, bsofa, dsofa, sep_wdw, sep_onset, sep_prop], axis=1)


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

    RUN_MANUAL = False
    RUN_TEST = True
    RUN_DATA = False


    if RUN_MANUAL:

        # -------------------
        # Create test samples
        # -------------------
        n, t, f = None, True, False

        p1 = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [n, n, t, n, n, n, n, n, n, n],
            [n, n, t, n, n, n, n, n, n, n],
            [1, 2, 3, 4, 5, 4, 3, 2, 1, 1],
            [n, n, 1, n, n, n, n, n, n, n], # pathogen a
            [n, n, 1, n, n, n, n, n, n, n]  # pathogen b
        ]).T

        p2 = np.array([
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [n, n, t, n, n, n, n, n, n, n],
            [n, n, n, n, t, n, n, n, n, n],
            [1, 1, 5, 6, 7, 1, 1, 4, 1, 1],
            [n, n, n, n, 1, n, n, n, n, n], # pathogen a
            [n, n, n, n, 0, n, n, n, n, n]  # pathogen b
        ]).T

        p3 = np.array([
            [3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [n, n, n, n, n, t, n, n, n, n],
            [n, n, t, n, n, n, n, n, n, n],
            [1, 1, 5, 1, 9, 1, 2, 1, 1, 1],
            [n, n, 1, n, n, n, n, n, n, n], # pathogen a
            [n, n, 0, n, n, n, n, n, n, n]  # pathogen b
        ]).T

        p4 = np.array([
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [n, t, n, n, n, n, n, n, n, n],
            [n, n, n, n, n, n, n, n, t, n],
            [1, 1, 5, 1, 9, 1, 2, 1, 1, 1],
            [n, n, n, n, n, n, n, n, 1, n], # pathogen a
            [n, n, n, n, n, n, n, n, 0, n]  # pathogen b
        ]).T

        p5 = np.array([
            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [t, n, n, n, n, n, n, n, t, n],
            [n, t, n, n, n, n, n, t, n, n],
            [1, 1, 5, 1, 9, 1, 2, 1, 1, 1],
            [n, 0, n, n, n, n, n, n, 1, n], # pathogen a
            [n, 0, n, n, n, n, n, n, 1, n]  # pathogen b
        ]).T


        p6 = np.array([
            [6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            [n, n, n, n, n, n, n, n, n, n],
            [n, n, n, n, n, n, n, n, n, n],
            [1, 1, 5, 1, 9, 1, 2, 1, 1, 1],
            [n, n, n, n, n, n, n, n, n, n], # pathogen a
            [n, n, n, n, n, n, n, n, n, n]  # pathogen b
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
            'sofa': float,
            'path1': int,
            'path2': int
        }

        # Create DataFrames
        df1 = pd.DataFrame(p1, columns=dtypes.keys())
        df2 = pd.DataFrame(m, columns=dtypes.keys())

        # Ensure they do not have range index
        df1.index = df1.index.values
        df2.index = df2.index.values

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
        si0 = suspected_infection(df1)
        si1 = propagate(r0, bck=1, fwd=2)
        si2 = propagate(r0, bck=2, fwd=3)

        # Compute with propagation
        si3 = suspected_infection(df1, prop=(2, 4))

        # Compute with group and propagation
        si4 = df2.groupby(by='stay_id').apply(
            suspected_infection, prop=(1,3),
            include_groups=False
        ).droplevel(0)

        # Compute si_lin (not working!)
        si5 = df2.groupby(by='stay_id').apply(
            si_lin2021, bck=1, fwd=1,
            include_groups=False
        ).droplevel(0)

        # ------------
        # si_moor_2023
        # ------------
        si6 = df2.groupby(by='stay_id').apply(
            si_moor2023,
            bck_abx=0, fwd_abx=2,
            bck_smp=0, fwd_smp=6,
            bck_si=4, fwd_si=2,
            include_groups=False
        ).droplevel(0)

        # Show
        lbl = 'Suspected Infection'
        print("\n%s\n%s\n%s" % ("=" * 80, lbl, "=" * 80))
        print("\n%s (manual propagation)" % lbl)
        print(pd.concat([df1, si0, si1, si2, si3], axis=1))
        print("\n%s (group propagation)" % lbl)
        print(pd.concat([df2, si4, si5, si6], axis=1))

        """
        # To see it split by patients
        aux = pd.concat([df2, r6], axis=1)
        for i,g in aux.groupby('stay_id'):
            print("\nStay: %s" % i)
            print(g)
        """


        # ------------------------------------------------
        # Compute bacteremia
        # ------------------------------------------------
        # Compute
        bac0 = bac_bahp2024(df1)

        # ------------
        # bac_bap2024
        # ------------
        bac6 = df2.groupby(by='stay_id').apply(
            bac_bahp2024, bck_bac=2, fwd_bac=2,
            pathogen_columns=['path1', 'path2'],
            include_groups=False
        ).droplevel(0)

        # Show
        lbl = 'Bacteremia'
        print("\n%s\n%s\n%s" % ("=" * 80, lbl, "=" * 80))
        print("\n%s (manual propagation)" % lbl)
        print(pd.concat([df1, bac0], axis=1))
        print("\n%s (group propagation)" % lbl)
        print(pd.concat([df2, bac6], axis=1))


        # -----------------------------------------------
        # Compute bloodstream infection
        # -----------------------------------------------
        # Compute
        bsi0 = bsi_bahp2024(df1)

        # bsi_bap2024
        # ------------
        bsi6 = df2.groupby(by='stay_id').apply(
            bsi_bahp2024, bck_abx=0, fwd_abx=24,
            bck_smp=0, fwd_smp=72,
            bck_si=48, fwd_si=24,
            bck_bac=2, fwd_bac=2,
            pathogen_columns=['path1', 'path2'],
            include_groups=False
        ).droplevel(0)

        # Show
        lbl = 'Bloodstream Infection'
        print("\n%s\n%s\n%s" % ("=" * 80, lbl, "=" * 80))
        print("\n%s (group propagation)" % lbl)
        print(pd.concat([df2, bsi6], axis=1))


        # -----------------------------------------------
        # Compute sepsis
        # -----------------------------------------------
        """
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
        """


        # ------------------
        # sep_moor2023
        # ------------------
        s5 = df2.groupby(by='stay_id').apply(
            sep_moor2023, bck_abx=0, fwd_abx=2,
            bck_smp=0, fwd_smp=6, bck_si=4, fwd_si=2,
            include_groups=False
        ).droplevel(0)

        # Show
        #print("\n%s\nSepsis\n%s" % ("="*80, "="*80))
        #print("\nSepsis (manual propagation)")
        #print(pd.concat([df1, s0, s1, s2], axis=1))
        #print("\nSepsis (auto propagation)")
        #print(pd.concat([df1, s3], axis=1))
        print("\nSepsis (group propagation)")
        print(pd.concat([df2, s5], axis=1))


    if RUN_TEST:

        # ------------------------------------------------
        # Test using fixture
        # ------------------------------------------------
        # Read data
        df = pd.read_csv('./fixtures/test_prep.csv')

        # -----------
        # bac_bap2024
        # ------------
        bac = df.groupby(by='stay_id').apply(
            bac_bahp2024, bck_bac=0, fwd_bac=2,
            pathogen_columns=None,
            include_groups=False
        ).droplevel(0)

        # ------------
        # si_moor2023
        # ------------
        #si = df.groupby(by='stay_id').apply(
        #    si_moor2023,
        #    bck_abx=0, fwd_abx=2,
        #    bck_smp=0, fwd_smp=2,
        #    bck_si=2, fwd_si=2,
        #    include_groups=False
        #).droplevel(0)

        # ------------
        # sep_moor2023
        # ------------
        # Compute with group and propagation
        sep = df.groupby(by='stay_id').apply(
            sep_moor2023, delta_sofa=2,
            bck_abx=0, fwd_abx=2,
            bck_smp=0, fwd_smp=2,
            bck_si=2, fwd_si=2,
            bck_sep=0, fwd_sep=2,
            include_groups=False
        ).droplevel(0)

        # Concatenate
        aux = pd.concat([df, bac, sep], axis=1)
        aux = aux.drop(columns=['Description'])

        print(aux.columns)

        # Compute mismatches
        r1 = aux.sol_sion.astype('boolean') != aux.si_on_moor2023
        r2 = aux.sol_sepon.astype('boolean') != aux.sep_on_moor2023
        r3 = aux.sol_bac1_on.astype('boolean') != aux.bac_on_bahp2024

        # Show
        lbl = 'Testing'
        print("\n%s\n%s\n%s" % ("=" * 80, lbl, "=" * 80))
        print("SI mismatches: %s" % r1.sum())
        print("SEP mismatches: %s" % r2.sum())
        print("BAC mismatches: %s" % r3.sum())

        # Show rows if needed
        #print(aux[r3])


    if RUN_DATA:

        # -----------------------------------------------
        # Example with real data
        # -----------------------------------------------
        # Libraries
        from pathlib import Path

        # Data
        DATAPATH = Path('C:\\Users\\kelda\\Desktop\\datasets\\ricu\\mac')

        # Whether debug
        DEBUG = True

        # Columns to visualise from loaded data.
        cols = ['stay_id', 'stay_time', 'abx', 'samp', 'sofa']

        # .. note: Use this variable to filter patients for which
        #          it would be interesting to review whether the
        #          computation of si, bac, bsi and sep is right.

        # Interesting ids
        stay_ids = [
            #3046540
            201006
        ]

        # Original data sets (export_data function)
        db = 'eicu_demo_0.5.6.parquet'
        #db = 'mimic_demo_0.5.6.parquet'
        #db = 'eicu_0.5.6.parquet'
        #db = 'hirid_0.5.6.parquet'
        #db = 'mimic_0.5.6.parquet'
        #db = 'aumc_0.5.6.parquet'
        #db = 'miiv_0.5.6.parquet'

        # Only just enough for labels (cs3pt)
        #db = 'eicu_demo_0.5.6_cs3pt_bahp.parquet'
        #db = 'mimic_demo_0.5.6_cs3pt_bahp.parquet'
        #db = 'eicu_0.5.6_cs3pt_bahp.parquet'
        #db = 'hirid_0.5.6_cs3pt_bahp.parquet'
        #db = 'mimic_0.5.6_cs3pt_bahp.parquet'
        #db = 'aumc_0.5.6_cs3pt_bahp.parquet'
        #db = 'miiv_0.5.6_cs3pt_bahp.parquet'

        # Only those with raw features (raw)
        #db = 'eicu_demo_ricu_min.parquet'
        #db = 'mimic_demo_ricu_min.parquet'
        #db = 'eicu_demo_cmb_0.5.6.parquet'
        #db = 'df_eicu_with_micro.parquet'
        #db = 'eicu_demo_ricu_cs3t_bahp.parquet'
        db = 'eicu_0.5.6_raw.parquet'
        db = 'eicu_demo_0.5.6_cs3t.parquet'


        src = 'mimic_demo'

        # Load merged
        db = '%s_0.5.6_mgd.parquet' % src

        # Show information
        print("\nDATAPATH: %s" % (DATAPATH/ db))

        # Load data
        df = pd.read_parquet(DATAPATH / db)
        df = df.rename(columns={
            'patientunitstayid': 'stay_id',
            'infusionoffset': 'stay_time',
            'labresultoofset': 'stay_time',
            'icustay_id': 'stay_id',
            'admissionid': 'stay_id',
            'start': 'stay_time',
            'startdate': 'stay_time',
            'culturetakenoffset': 'samp'
        })

        # Filter interesting ids
        #df = df[df.stay_id.isin(stay_ids)]

        # See generic info
        print("\nSample:")
        print(df)
        print("\nColumns:")
        print(df.columns.tolist())
        print("\nIs <samp> in columns?:", 'samp' in df.columns.tolist())

        # See some statistics
        if 'sep3' in df:
            print("Sep3: %s" % df.sep3.sum())
        if 'samp' in df:
            print('Samp: %s' % df.samp.sum())
        if 'abx' in df:
            print("abx: %s" % df.abx.sum())


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


        if DEBUG:
            print("\nDF:")
            print(df)


        # -----------
        # bac_bap2024
        # ------------
        bac = df.groupby(by='stay_id').apply(
            bac_bahp2024, bck_bac=0, fwd_bac=24,
            pathogen_columns=None,
            include_groups=False
        ).droplevel(0)


        # ------------
        # si_moor2023
        # ------------
        si = df.groupby(by='stay_id').apply(
            si_moor2023,
            bck_abx=0, fwd_abx=24,
            bck_smp=0, fwd_smp=72,
            bck_si=48, fwd_si=24,
            include_groups=False
        )

        print("%s\n%s\n%s" % ("="*80, 'SI', "="*80))
        print("\nResult:")
        print(si)
        print("\nCount:")
        print(si.sum(axis=0))
        print("\nWith onset:")
        print(si[si.si_ons_moor2023 == True])

        # ------------
        # sep_moor2023
        # ------------
        # Compute with group and propagation
        sep = df.groupby(by='stay_id').apply(
            sep_moor2023,
            include_groups=False
        ).droplevel(0)

        # Concatenate
        aux = pd.concat([df, bac, sep], axis=1)

        print("\nResult:")
        print(aux)
        print("\nCount:")
        print(sep.sum(axis=0))
        print("\nWith onset:")
        print(aux[aux.sep_on_moor2023 == True])

        # Save
        #aux.to_parquet(DATAPATH / ('%s_0.5.6_lbl.parquet' % src))


    import sys
    sys.exit()
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

