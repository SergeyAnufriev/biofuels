from rdkit import Chem
import numpy as np
from pyADA import Similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

sim = Similarity()


def finger_print(data):
    fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)).ToList() for x in data['SMILES'].values]

    return np.array(fps)


def mean_tanimato(smile, X_train):
    x = Chem.RDKFingerprint(Chem.MolFromSmiles(smile)).ToList()
    X_train_fingerprint = finger_print(X_train)
    s = 0

    for i in range(len(X_train)):
        s += sim.tanimoto_similarity(x, X_train_fingerprint[i, :])

    return s / len(X_train)


'''https://pubs.acs.org/doi/10.1021/acs.jcim.8b00597'''


def exp_td(smile, X_train):
    x = Chem.RDKFingerprint(Chem.MolFromSmiles(smile)).ToList()

    X_train_fingerprint = finger_print(X_train)

    scores = []

    for i in range(len(X_train)):
        scores.append(sim.tanimoto_similarity(x, X_train_fingerprint[i, :]))

    scores = np.array(scores)

    return np.sum(np.exp(-1 * (3 * scores) / (1 - scores)))


def Leverage(x_in, X_train):
    inner_mat = np.matmul(X_train.T, X_train) ** (-1)

    return np.matmul(np.matmul(x_in.T, inner_mat), x_in)


scalar = MinMaxScaler()


class Partition:
    def __init__(self, X_train, X_test, errors, by_type, fractions, **kwargs):

        self.X_train = X_train
        self.X_test = X_test
        self.errors = errors
        self.by_type = by_type
        self.fractions = fractions

        if by_type == 'OneClassSVM':
            self.model = OneClassSVM()
            if len(kwargs) != 0:
                self.model.set_params(**kwargs)
        elif by_type == 'IsolationForest':
            self.model = IsolationForest()
            if len(kwargs) != 0:
                self.model.set_params(**kwargs)
        else:
            self.model = None

    def binary_classifier(self):

        error_break_down = {}
        for of in self.fractions:
            if self.by_type == 'OneClassSVM':
                self.model.nu = of
            else:
                self.model.contamination = of
            self.model.fit(self.X_train.values[:, 1:])
            idx = np.where(self.model.predict(self.X_test[self.X_train.columns[1:]]) > 0)[0]
            error_break_down[of] = {'error': self.errors[idx], 'SMILES': self.X_test.SMILES.values[idx]}

        return error_break_down

    def check_inverse(self):

        X = np.matmul(self.X_train.values[:, 1:].T, self.X_train.values[:, 1:]).astype('float')
        if np.linalg.det(X) == 0:
            return False
        else:
            return True

    def distance_model(self):

        if self.by_type == 'Tanimoto':
            dist_ = self.X_test['SMILES'].apply(lambda x: mean_tanimato(x, self.X_train)).values
        else:
            if self.check_inverse() is True:
                x = self.X_test[self.X_train.columns].values[:, 1:]
                dist_ = [Leverage(x[i, :], self.X_train.values[:, 1:]) for i in range(len(self.X_test))]
            else:
                dist_ = None
        return dist_

    def distance_bins(self):

        error_break_down = {}
        dist_ = self.distance_model()
        if dist_ is None:
            return None
        else:
            for x in self.fractions:
                idx = np.where(dist_<x)[0]
                error_break_down[x] = {'error': self.errors[idx], 'SMILES': self.X_test.SMILES.values[idx]}

        return error_break_down

    def calculate(self):
        if self.by_type == 'IsolationForest' or self.by_type == 'OneClassSVM':
            return self.binary_classifier()
        else:
            return self.distance_bins()
