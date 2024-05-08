# Libraries
import pandas as pd

from utils import hello
from utils import df_overview
from pathlib import Path

# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

d = {
    'id': {},
    'Ward Glucose': {},
    'Haemoglobin': {},
    'Mean cell volume, blood': {},
    'White blood cell count, blood': {
        'mimic_hosp': {
            'itemid': [51753, 51754, 51301]
        }
    },
    'Haematocrit': {},
    'Platelets' : {
        'mimic_hosp': {
            'itemid': [51240]
        }
    },
    'Urea level, blood': {},
    'Creatinine': {
        'mimic_hosp': {
            'itemid': [50912, 52541]
        }
    },
    'Sodium': {},
    'Potassium': {},
    'Lymphocytes': {
        'mimic_hosp': {
            'itemid': [51536, 51133, 52764, 51688 ]
        }
    },
    'Neutrophils': {
        'mimic_hosp': {
            'itemid': [51695, 51256, 51537, 52070]
        }
    },
    'C-Reactive Protein': {
        'mimic_hosp': {
            'itemid': [50889]
        },
    },
    'Eosinophils': {
        'mimic_hosp': {
            'itemid': [51199, 51114, 51200, 52068]
        }
    },
    'Alkaline Phosphatase': {
        'mimic_hosp': {
            'itemid': [50863]
        }
    },
    'Albumin': {
        'mimic_hosp': {
            'itemid': [51542]
        }
    },
    'Alanine Transaminase': {
        'mimic_hosp': {
            'itemid': [50861]
        }
    },
    'Bilirubin': {
        'mimic_hosp': {
            'itemid': [50883, 50884, 50885]
        }
    },
    'Total Protein': {},
    'Fibrinogen (clauss)': {
        'mimic_hosp': {
            'itemid': [52110, 51621, 52111]
        }
    },
    'Glucose POCT Strip Blood': {
        'mimic_hosp': {
            'itemid': [50809, 50931, 52564]
        }
    },
    'Ferritin': {
        'mimic_hosp': {
            'itemid': [50924]
        }
    },
    'D-Dimer': {
        'mimic_hosp': {}
    },
    'Ward Lactate': {
        'mimic_hosp': {
            'itemid': [50813, 52437]
        }
    },
    'age': {},
    'sex': {},
    'SARS CoV-2 RNA': {
        'mimic_hosp': {
            'itemid': [51847, 51846]
        }
    },
    'pathogenic': {}
}


def get_feature_itemids(dataset='mimic_hosp',
                        column='itemid'):
    """Creates map with <id: name>"""
    l = {}
    for k,v in d.items():
        if not dataset in v:
            continue
        if not column in v[dataset]:
            continue
        for id in v[dataset][column]:
            l[id] = k
    return l


COLUMNS = [
    'subject_id',
    'hadm_id',
    'specimen_id',
    'itemid',
    #'label',
    #'fluid',
    #'category',
    'value'
]
"""
labevent_id          int64
subject_id           int64
hadm_id            float64
specimen_id          int64
itemid               int64
charttime           object
storetime           object
value               object
valuenum           float64
valueuom            object
ref_range_lower    float64
ref_range_upper    float64
flag                object
priority            object
comments            object
label               object
fluid               object
category            object
loinc_code          object
title               objec
"""

# ---------------------------------------------
# Configuration
# ---------------------------------------------
# Path
DATA_ROOT = Path('/Users/cbit/Desktop/datasets/')
DATA_PATH = DATA_ROOT / 'physionet-mimic/mimic-iv-0.4'


df_patients =  pd.read_csv(DATA_PATH / 'core' / 'patients.csv', nrows=100)

print(df_patients)


# ---------------
# Pathology
# ---------------
# Load pathology data and id mappings
df_labdict = pd.read_csv(DATA_PATH / 'hosp' / 'd_labitems.csv')
df_labevents = pd.read_csv(DATA_PATH / 'hosp' / 'labevents.csv',
    usecols=COLUMNS) #, nrows=10000)

# Merge
df_labevents = df_labevents.merge(df_labdict,
    how='left', left_on='itemid', right_on='itemid')


def title(x):
    return '%s, %s, %s (%s)' % (x.label, x.fluid, x.category, x.itemid)


df_labevents['title'] = df_labevents.apply(
    title, axis=1
)

print(df_labevents)
hello('bernard')

df_overview(df_labevents)

# Get lab id to name mapping
id2name = get_feature_itemids('mimic_hosp')

rename = get_feature_itemids('mimic_hosp')
df_aux = df_labevents[df_labevents.itemid.isin(rename.keys())]
print(rename)

print(df_aux)

# Pivot
df_pivot = df_aux.pivot(
    index=['subject_id', 'hadm_id', 'specimen_id'],
    columns='title',
    values=['value'])
print(df_pivot)

df_pivot.columns = df_pivot.columns.droplevel()
df_pivot = df_pivot.reset_index(drop=True)
print(df_pivot)


from ydata_profiling import ProfileReport
#df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])
profile = ProfileReport(df_pivot, title="Profiling Report")
profile.to_file("mimic.html")

"""
def create_labevents():
    # Load
    df_labdict = pd.read_csv(source / 'd_labitems.csv')
    df_labevents = pd.read_csv(source / 'labevents.csv' , nrows=100000)

    # Combine
    df_labevents = df_labevents.merge(df_labdict,
        how='left', left_on='itemid', right_on='itemid')


    def title(x):
        return '%s, %s, %s (%s)' % (x.label, x.fluid, x.category, x.itemid)

    df_labevents['title'] = df_labevents.apply(
        title, axis=1
    )

    # Add new column
    #df_labevents['title'] = \
    #    df_labevents[['label', 'fluid', 'category', 'itemid']]\
    #        .agg('-'.join, axis=1)

    # Pivot
    df_pivot = df_labevents.pivot(
        index=['subject_id', 'hadm_id', 'specimen_id'],
        columns='title',
        values=['value'])
    print(df_pivot)

    return df_pivot


df = create_labevents()
df.columns = df.columns.droplevel()
df = df.reset_index(drop=True)
print(df)

#df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])
profile = ProfileReport(df, title="Profiling Report")
profile.to_file("your_report.html")
"""
