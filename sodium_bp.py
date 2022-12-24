import pandas as pd
import sqlite3
import sqlalchemy
import numpy as np
import os
import json
import copy


def generate_data(path, N):
    age = np.random.normal(65, 5, size=(N, 1))
    sodium = age / 18. + np.random.normal(0, 1, size=(N, 1))
    sodium = (sodium >= 3.5).astype('float')
    print(f"The number of treated group is {sodium.sum()}, and the number of control group is is {N - sodium.sum()}.\n")
    blood_pressure = 1.05 * sodium + 2.00 * age + np.random.normal(0, 1, size=(N, 1))
    protein = 2.00 * blood_pressure + 2.80 * sodium + np.random.normal(0, 1, size=(N, 1))
    data = {}
    data['age'] = age.tolist()
    data['protein'] = protein.tolist()
    data['sodium'] = sodium.tolist()
    data['blood_pressure'] = blood_pressure.tolist()
    with open(f"{path}/data_dict.json", "w+") as f:
        json.dump(data, f, indent=4)
        print(f"Saving generated data to {path}/data_dict.json\n")
    return data


def main():
    print("The current working directory is:", os.getcwd())
    N = 1000
    eps = 1e-6
    np.random.seed(0)
    dataset_dir = '../brady_neal/dyy275_data'
    data = generate_data(dataset_dir, N)

    T = np.array(data['sodium'])
    Y_1 = np.array(data['blood_pressure'])[T == 1][:, None]
    Y_0 = np.array(data['blood_pressure'])[T == 0][:, None]
    Y = np.array(data['blood_pressure'])

    # The adjustment set is {}
    ATE = Y_1.mean() - Y_0.mean()
    print(f"ATE with no adjustment set is {ATE}.\n")
    
    # The adjustment set is {Age}
    X = np.array(data['age'])
    XT = np.append(X, T, axis=1)
    XT_with_bias = np.append(XT, np.array([[1] * len(XT)]).T, axis=1)
    XT_1 = copy.deepcopy(XT_with_bias)
    XT_1[:, 1] = 1
    XT_0 = copy.deepcopy(XT_with_bias)
    XT_0[:, 1] = 0
    coeff = np.linalg.inv(XT_with_bias.T @ XT_with_bias + eps * np.eye(3)) @ XT_with_bias.T @ Y
    ATE = (XT_1 @ coeff).mean() - (XT_0 @ coeff).mean()
    print(f"ATE with Age as adjustment set is {ATE}.\n")

    # The adjustment set is {Age, protein}
    XT = np.concatenate((np.array(data['age']), np.array(data['protein']), T), axis=1)
    XT_with_bias = np.append(XT, np.array([[1] * len(XT)]).T, axis=1)
    XT_1 = copy.deepcopy(XT_with_bias)
    XT_1[:, 2] = 1
    XT_0 = copy.deepcopy(XT_with_bias)
    XT_0[:, 2] = 0
    coeff = np.linalg.inv(XT_with_bias.T @ XT_with_bias + eps * np.eye(4)) @ XT_with_bias.T @ Y
    ATE = (XT_1 @ coeff).mean() - (XT_0 @ coeff).mean()
    print(f"ATE with Age and protein as adjustment set is {ATE}.\n")
    return 0


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
