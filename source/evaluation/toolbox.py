import numpy as np
import pandas as pd

def DictCompare(W, X, op = 'intersection'):
    if op == 'intersection':
        shared_W_keys = list(W.keys() & X.keys())
        shared_W_keys.sort()
        W = DictToList(W, shared_W_keys)
        X = DictToList(X, shared_W_keys, WorX='X')
        return shared_W_keys, W, X # keys and corresponding values
    if op == 'extraW':
        extra_W_keys = np.sort(np.array(list(W.keys() - X.keys())))
        extra_W_values = DictToList(W, extra_W_keys)
        return extra_W_keys, extra_W_values

def DictToList(dict, indexList = None, WorX = 'weight'):
    df = pd.DataFrame(dict)
    if indexList is not None:
        df_sel = df[indexList]
        if WorX != 'weight':
            return df_sel.values.reshape(1, -1)
        else:
            return df_sel.values.reshape(-1, 1)
    else:
        if WorX != 'weight':
            return df.values.reshape(1, -1)
        else:
            return df.values.reshape(-1, 1)

def MatrixInDict(matrix, dict):
    matrixTemp = matrix.copy()
    '''Always take the full feature space as the dimension of mapped vector'''
    for (r,row) in enumerate(matrix):
        for (c,col) in enumerate(row):
            if dict.get(r) is not None:
                key_new_feature = dict.get(r)
                if key_new_feature is not None:
                    key_new_to_all = key_new_feature.get(c)
                    matrixTemp[r, c] = key_new_to_all
    return matrixTemp