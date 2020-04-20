import pandas as pd
import numpy as np

CLASS = 2


def read_csv():
    x_train = pd.read_csv("x_train.csv").values
    y_train = pd.read_csv("y_train.csv").values[:, 0]
    x_test = pd.read_csv("x_test.csv").values
    y_test = pd.read_csv("y_test.csv").values[:, 0]
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    return x_train, y_train, x_test, y_test


def compute_mean(x_train, y_train):
    class_mean = np.empty((0, x_train.shape[1]), float)
    for class_idx in range(CLASS):
        match_idx = np.where(y_train == class_idx)
        tmp_mean = np.mean(x_train[match_idx], axis=0)
        class_mean = np.vstack([class_mean, tmp_mean])
        print("mean vector of class {}: {}".format(class_idx, tmp_mean))
    return class_mean


def compute_withinclass(x_train, y_train, class_mean):
    within_class = np.zeros([x_train.shape[1], x_train.shape[1]])
    for data_idx in range(x_train.shape[0]):
        dist = np.subtract(x_train[data_idx], class_mean[y_train[data_idx]]).reshape(x_train.shape[1], 1)
        within_class += np.matmul(dist, dist.T)
    print("Within-class scatter matrix SW: {}".format(within_class))
    return within_class


def compute_betweenclass(class_mean):
    dist = np.subtract(class_mean[0], class_mean[1]).reshape(class_mean.shape[1], 1)
    between_class = np.matmul(dist, dist.T)
    print("Between-class scatter matrix SB: {}".format(between_class))
    return between_class


def compute_eigen():
    print()


def compute_accuracy():
    print()


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = read_csv()
    class_mean = compute_mean(x_train, y_train)
    within_class = compute_withinclass(x_train, y_train, class_mean)
    between_class = compute_betweenclass(class_mean)
    compute_eigen()
    compute_accuracy()