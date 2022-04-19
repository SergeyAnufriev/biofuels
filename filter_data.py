import pandas as pd
from sklearn.feature_selection import VarianceThreshold
import numpy as np


def drop_miss_column(data: pd.DataFrame) -> pd.DataFrame:
    """Remove columns with NaN values"""

    missing_val_cols = data.columns[data.isnull().any()]
    x = data.copy()
    df_ = x.drop(missing_val_cols, axis=1)

    return df_


def low_var(data: pd.DataFrame, threshold=0.01) -> pd.DataFrame:
    """This function removes low variance columns in dataframe
    threshold is minimum allowed variance in data columns"""

    selector = VarianceThreshold(threshold)
    selector.fit(data)
    return data[data.columns[selector.get_support(indices=True)]]


def high_corr(data: pd.DataFrame, threshold=0.95) -> pd.DataFrame:
    """Removes correlated columns in data frame with pearson
    correlation higher then threshold"""

    cor_matrix = data.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    cols = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    x = data.copy()

    return x.drop(cols, axis=1)


def high_skewed(data: pd.DataFrame, threshold=4) -> pd.DataFrame:
    """Retain dataframe columns with skewness lower then threshold"""

    return data[data.columns[abs(data.skew()) < threshold]]


def compose(*function_param):
    def inner(arg):
        for f, param in function_param:
            arg = f(arg, param)
        return arg

    return inner


class Select_descriptors:
    """This class combines the above filtering steps"""

    def __init__(self, var_: float, cor_: float, skew_: float,wrapper:object):
        self.var_ = var_
        self.cor_ = cor_
        self.skew_ = skew_
        self.transforms = [(x, y) for x, y in zip([low_var, high_corr, high_skewed], [var_, cor_, skew_]) if
                           y is not None]

        self.wrapper   = wrapper
        self.variables = None

    def transform(self, *args) :
        """Input descriptors data frame with smiles column"""
        func = compose(*self.transforms)
        x    = args[0].copy()
        descriptors = func(x.drop('SMILES', axis=1))

        if self.wrapper == None:
            self.variables = descriptors.columns

            return pd.concat([x['SMILES'], descriptors], axis=1)
        else:
            y = args[1].reset_index(drop=True)
            desc_sel = self.wrapper.fit_transform(descriptors,y)
            self.variables = descriptors.columns[self.wrapper.get_support()]

            return pd.concat([x['SMILES'],pd.DataFrame(desc_sel, columns=self.variables)],axis=1)



