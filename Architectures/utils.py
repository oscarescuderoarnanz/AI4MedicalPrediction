import pandas as pd
import numpy as np

def dataframeToTensor(df, timeStepLength):
    _, idx = np.unique(df.idx, return_index=True)
    listPatients = np.array(df.idx)[np.sort(idx)]

    index = df.index
    for i in range(len(listPatients)):
        df_trial = df[df.idx == listPatients[i]]

        if i == 0:
            X = np.array(df_trial)
            X = X.reshape(1, timeStepLength, df.shape[1])
        else:
            X_2 = np.array(df_trial)
            X_2 = X_2.reshape(1, timeStepLength, df.shape[1])
            X = np.append(X, X_2, axis=0)


    X = np.delete(X, [50, 51], axis=2)
    
    return X