import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def vocabulary(df):
    s = set()
    d = {}
    rd = {}
    for r in range(df.shape[0]):
        for w in df.iloc[r, 0].split():
            s.add(w)
        df.iloc[r, 0] = df.iloc[r, 0]
    for i, w in enumerate(s):
        d[w] = i
        rd[i] = w
    return d, rd, list(s)


def lower_case(df):
    for r in range(df.shape[0]):
        df.iloc[r, 0] = df.iloc[r, 0].lower()
    return df


def num_vec(data, index_d):
    data_vec = pd.DataFrame()
    for r in range(data.shape[0]):
        vec = []
        for s in data.iloc[r, 0].split():
            vec.append(index_d[s])
        data_vec = data_vec.append(pd.DataFrame([[vec]]))
    data_vec = data_vec.reset_index(drop=True)
    data = data.reset_index(drop=True)
    data_vec["label"] = data["label"]
    return data_vec


