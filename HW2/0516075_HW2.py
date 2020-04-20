import pandas as pd
import numpy as np

CLASS = 2
K = 7


def read_csv():
    x_train = pd.read_csv("x_train.csv").values
    y_train = pd.read_csv("y_train.csv").values[:, 0]
    x_test = pd.read_csv("x_test.csv").values
    y_test = pd.read_csv("y_test.csv").values[:, 0]
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


def compute_eigen(A):
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    idx = eigenvalues.argsort()[::-1]                          # sort largest
    weight = eigenvectors[:,idx][:,:1]
    print("Fisher’s linear discriminant: {}".format(weight))
    return weight


def k_nearest_neighbor(x_train, y_train, x_test):
    y_pred = []
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    for test_idx in range(test_size):
        all_dist = np.zeros(train_size)
        for train_idx in range(train_size):
            dist = np.sqrt(np.sum((x_test[test_idx] - x_train[train_idx]) ** 2))
            all_dist[train_idx] = dist
        sort_idx = all_dist.argsort()
        neighbor = list(y_train[sort_idx][:K])
        prediction = max(set(neighbor), key=neighbor.count)
        y_pred.append(prediction)
    return y_pred


def compute_accuracy(y_pred, y_test):
    correct = 0
    for i in range(len(y_pred)):
        # print("Prediction: {}, Answer: {}".format(y_pred[i], y_test[i]))
        if y_pred[i] == y_test[i]:
            correct += 1
    print("Accuracy of test-set {}".format(correct / len(y_pred)))


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = read_csv()
    class_mean = compute_mean(x_train, y_train)
    within_class = compute_withinclass(x_train, y_train, class_mean)
    between_class = compute_betweenclass(class_mean)
    weight = compute_eigen(np.matmul(np.linalg.pinv(within_class), between_class))
    lower_dimension_train = np.matmul(x_train, weight)
    lower_dimension_test = np.matmul(x_test, weight)
    y_pred = k_nearest_neighbor(lower_dimension_train, y_train, lower_dimension_test)
    compute_accuracy(y_pred, y_test)