import pandas as pd
import os
import numpy as np


def get_ihdp_data():
    # if the local file already exists
    if os.path.exists('ihdp.csv'):
        data = pd.read_csv('ihdp.csv')
        if data.shape[0] == 3735 and data.shape[1] == 30:
            return data.to_numpy()
        else:
            pass

    # else, fetch the data from AMLab-Amsterdam
    for j in range(1, 6):
        data = pd.read_csv(
            f"https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_{j}.csv", header=None)
        col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1", ]
        for i in range(1, 26):
            col.append("x" + str(i))
        data.columns = col
        data = data.astype({"treatment": 'bool'}, copy=False)
        data.head()
        if j == 1:
            data_all = data
        else:
            data_all = pd.concat([data_all, data], axis=0)
    data_all.to_csv('ihdp.csv', index=False)

    # Convert pandas dataframe to numpy array
    return data_all.to_numpy()


def load_and_format_covariates_ihdp(data):
    binfeats = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
    contfeats = [i for i in range(25) if i not in binfeats]

    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5:]
    perm = binfeats + contfeats
    x = x[:, perm]
    return x.astype('float32')


def load_all_other_crap(data):
    t, y, y_cf = data[:, 0], data[:, 1][:, None], data[:, 2][:, None]
    mu_0, mu_1, x = data[:, 3][:, None], data[:, 4][:, None], data[:, 5:]
    return t.reshape(-1, 1).astype('float32'), y.astype('float32'), \
           y_cf.astype('float32'), mu_0.astype('float32'), mu_1.astype('float32')


def main():
    get_ihdp_data()
    return


if __name__ == '__main__':
    main()

