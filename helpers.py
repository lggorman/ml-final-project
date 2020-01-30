from math import log2
import numpy as np

def log_transform(s):
    for row in range(s.shape[0]):
        for col in range(s.shape[1]):
            if s[row][col] != 0:
                s[row][col] = log2(s[row][col])
    return s

# Doesn't seem to help
def trim_sparse_features(s, features_to_keep=None):
    new = []
    print(s.shape[1])
    if not features_to_keep:
        print('this?')
        features_to_keep = []
        for col in range(s.shape[1]):
            nonzero = [i for i in s[:,col] if i > 0]
            if len(nonzero) > 1:
                features_to_keep.append(col)
        print('keeping', len(features_to_keep))
    for row in range(s.shape[0]):
        new.append([])
        for col in range(s.shape[1]):
            if col in features_to_keep:
                new[row].append(s[row][col])
    return np.array(new), features_to_keep