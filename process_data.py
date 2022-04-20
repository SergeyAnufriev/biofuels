from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class Scalar(BaseEstimator, TransformerMixin):
    def __init__(self, type_):
        if type_ == 'minmax':
            self.sc = MinMaxScaler()
        elif type_ == 'rubust':
            self.sc = RobustScaler()
        else:
            self.sc = StandardScaler()

    def fit(self, x):
        self.sc.fit(x[x.columns[1:]])

    def transform(self, x):
        x_ = x.copy()
        descriptors = pd.DataFrame(self.sc.transform(x_[x_.columns[1:]]), columns=x_.columns[1:])
        smiles = x['SMILES'].reset_index(drop=True)

        return pd.concat([smiles, descriptors], axis=1)
