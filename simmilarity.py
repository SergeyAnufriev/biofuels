from rdkit import Chem
import numpy as np
from pyADA import Similarity


sim = Similarity()

def finger_print(data):

    fps = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)).ToList() for x in data['SMILES'].values]

    return np.array(fps)


def mean_tanimato(smile,X_train):

    x = Chem.RDKFingerprint(Chem.MolFromSmiles(smile)).ToList()

    X_train_fingerprint = finger_print(X_train)
    s = 0

    for i in range(len(X_train)):

        s += sim.tanimoto_similarity(x,X_train_fingerprint[i,:])

    return s/len(X_train)

'''https://pubs.acs.org/doi/10.1021/acs.jcim.8b00597'''

def exp_td(smile,X_train):

    x = Chem.RDKFingerprint(Chem.MolFromSmiles(smile)).ToList()

    X_train_fingerprint = finger_print(X_train)

    scores = []

    for i in range(len(X_train)):

        scores.append(sim.tanimoto_similarity(x,X_train_fingerprint[i,:]))

    scores = np.array(scores)

    return np.sum(np.exp(-1 * (3 * scores) / (1 - scores)))


def Leverage(x_in, X_train):

    inner_mat = np.matmul(X_train.T,X_train)**(-1)

    return np.matmul(np.matmul(x_in.T,inner_mat),x_in)


