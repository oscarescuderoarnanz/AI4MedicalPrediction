def hello(name):
    print("Hello %s" % name)

def df_overview(df):
    """Overview of DataFrame."""
    print("\nColumns:")
    print(df.columns)
    print("\nDtypes:")
    print(df.dtypes)
    print("\nRows:")
    print(df)