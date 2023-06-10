import numpy as np
import torch
from dragonnet import DragonNet
from dataset import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess(t, y_unscaled, x):
    y_mean = y_unscaled.mean()
    y_std = y_unscaled.std()
    y = (y_unscaled - y_mean) / y_std

    test_size = 0.2
    train_index, test_index = train_test_split(np.arange(x.shape[0]), test_size=test_size)
    if test_size == 0:
        test_index = train_index

    x_train, x_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    t_train, t_test = t[train_index], t[test_index]

    return y_mean, y_std, train_index, test_index, x_train, y_train, t_train, x_test, y_test, t_test


def main():
    data = get_ihdp_data()
    x = load_and_format_covariates_ihdp(data)
    t, y, y_cf, mu_0, mu_1 = load_all_other_crap(data)
    mean, std, train_index, test_index, x_train, y_train, t_train, x_test, y_test, t_test = preprocess(t, y, x)

    """
    Here is to load your own dataset
    x_train = ...
    y_train = ...
    t_train = ...
    """

    # Train the Dragon Net
    model = DragonNet(x_train.shape[1], epochs=5, learning_rate=1e-3)
    model.fit(x_train, y_train, t_train)

    # Predict
    y0_pred_test, y1_pred_test, t_pred_test, _ = model.predict(x_test)
    y0_hat = y0_pred_test * std + mean
    y1_hat = y1_pred_test * std + mean
    t_pred_test = (t_pred_test > 0.5).numpy().astype(float)
    ATE_pred_test = (y1_hat - y0_hat).mean()
    # True ATE is mu1 - mu0
    ATE_true_test = (data[test_index, 4] - data[test_index, 3]).mean()
    print(f"The true ATE on test set is {ATE_true_test}")
    print(f"The predicted ATE on test set is {ATE_pred_test}\n\n")
    treat_acc_test = np.equal(t_pred_test, t_test).mean()
    print(f"The predicted accuracy of treatment on test set is {treat_acc_test}\n\n")

    y0_pred_train, y1_pred_train, t_pred_train, _ = model.predict(x_train)
    y0_hat = y0_pred_train * std + mean
    y1_hat = y1_pred_train * std + mean
    t_pred_train = (t_pred_train > 0.5).numpy().astype(float)
    ATE_pred_train = (y1_hat - y0_hat).mean()
    # True ATE is mu1 - mu0
    ATE_true_train = (data[train_index, 4] - data[train_index, 3]).mean()
    print(f"The true ATE on train set is {ATE_true_train}")
    print(f"The predicted ATE on train set is {ATE_pred_train}\n\n")
    treat_acc_train = np.equal(t_pred_train, t_train).mean()
    print(f"The predicted accuracy of treatment on train set is {treat_acc_train}\n\n")
    return


if __name__ == '__main__':
    main()