# Libraries
import pandas as pd

from pathlib import Path
from ydata_profiling import ProfileReport
#df = pd.DataFrame(np.random.rand(100, 5), columns=["a", "b", "c", "d", "e"])

#profile = ProfileReport(df_pivot, title="Profiling Report")
#profile.to_file("mimic.html")


# Define paths.
data_path = Path('/Users/cbit/Desktop/repositories/github/multicenter-sepsis/data-export')
outp_path = Path('./outputs/reports')

# Create folder if it does not exist
outp_path.mkdir(parents=True, exist_ok=True)

# Default columns
cols = ['stay_id', 'stay_time']

# Loop datasets creating reports
for i,p in enumerate(data_path.glob('*.parquet')):
    # Information.
    print("%2s. Loading... <%s>" % (i, p))

    # Read dataset and filter
    df = pd.read_parquet(p)
    cols = ['stay_id', 'stay_time']
    df = df[cols + [c for c in df.columns if '_raw' in c]]

    # Create profile report
    profile = ProfileReport(df,
        minimal=True, title="Report - %s" % p.stem)
    profile.to_file(outp_path / ("report_%s.html" % p.stem))