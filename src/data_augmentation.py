# data_augmentation.py

from imblearn.over_sampling import SMOTE
import pandas as pd

def augment_data_with_smote(X, y):
    smote = SMOTE(sampling_strategy='auto')
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res
