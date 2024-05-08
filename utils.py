def dataframeToTensor(df, y, eliminateColumn, columns, timeStepLength):
    _, idx = np.unique(df.Admissiondboid, return_index=True)
    listPatients = np.array(df.Admissiondboid)[np.sort(idx)]

    index = df.index
    y = y.reindex(index)
    y = y.drop_duplicates(subset="Admissiondboid")

    for i in range(len(listPatients)):
        df_trial = df[df.Admissiondboid == listPatients[i]]
        if eliminateColumn:
            df_trial = df_trial.drop(columns=columns)
        if i == 0:
            X = np.array(df_trial)
            X = X.reshape(1, timeStepLength, df.shape[1] - len(columns))
        else:
            X_2 = np.array(df_trial)
            X_2 = X_2.reshape(1, timeStepLength, df.shape[1] - len(columns))
            X = np.append(X, X_2, axis=0)
            
    return X, y