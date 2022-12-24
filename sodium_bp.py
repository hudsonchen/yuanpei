import pandas as pd
import sqlite3
import sqlalchemy
import numpy as np
import os
import json


def generate_data(path, N):
    age = np.random.randint(5, 65, size=(N, 1))
    sodium = age / 18. + np.random.normal(0, 1, size=(N, 1))
    sodium = (sodium >= 2.0).astype('float')
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

    temp = np.array(data['sodium'])
    Y_1 = np.array(data['blood_pressure'])[temp == 1][:, None]
    Y_0 = np.array(data['blood_pressure'])[temp == 0][:, None]

    # The adjustment set is {}
    ATE = Y_1.mean() - Y_0.mean()
    print(f"ATE with no adjustment set is {ATE}.\n")
    
    # The adjustment set is {Age}
    W = np.array(data['age'])[:, None]
    W_1 = np.array(data['age'])[temp == 1][:, None]
    W_1_with_bias = np.append(W_1, np.array([[1] * len(W_1)]).T, axis=1)
    W_0 = np.array(data['age'])[temp == 0][:, None]
    W_0_with_bias = np.append(W_0, np.array([[1] * len(W_0)]).T, axis=1)
    coeff_1 = np.linalg.inv(W_1_with_bias.T @ W_1_with_bias + eps * np.eye(2)) @ W_1_with_bias.T @ Y_1
    coeff_0 = np.linalg.inv(W_0_with_bias.T @ W_0_with_bias + eps * np.eye(2)) @ W_0_with_bias.T @ Y_0
    ATE = (coeff_1[0] * W + coeff_1[1]).mean() - (coeff_0[0] * W + coeff_0[1]).mean()
    print(f"ATE with Age as adjustment set is {ATE}.\n")

    ## The adjustment set is {Age, protein}
    Z = np.array(data['protein'])[:, None]
    Z_1 = np.array(data['protein'])[temp == 1][:, None]
    ZW_1_with_bias = np.concatenate((W_1, Z_1, np.array([[1] * len(Z_1)]).T), axis=1)
    Z_0 = np.array(data['protein'])[temp == 0][:, None]
    ZW_0_with_bias = np.concatenate((W_0, Z_0, np.array([[1] * len(Z_0)]).T), axis=1)
    coeff_1 = np.linalg.inv(ZW_1_with_bias.T @ ZW_1_with_bias + eps * np.eye(3)) @ ZW_1_with_bias.T @ Y_1
    coeff_0 = np.linalg.inv(ZW_0_with_bias.T @ ZW_0_with_bias + eps * np.eye(3)) @ ZW_0_with_bias.T @ Y_0
    ATE = (coeff_1[0] * W + coeff_1[1] * Z + coeff_1[2]).mean() - (coeff_0[0] * W + coeff_0[1] * Z + coeff_0[2]).mean()
    print(f"ATE with Age and protein as adjustment set is {ATE}.\n")
    return 0


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
