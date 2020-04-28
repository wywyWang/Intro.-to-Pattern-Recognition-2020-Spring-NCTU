import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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
        dist = np.subtract(x_train[data_idx], class_mean[y_train[data_idx]])
        dist = dist.reshape(x_train.shape[1], 1)
        within_class += np.matmul(dist, dist.T)
    print("Within-class scatter matrix SW: {}".format(within_class))
    return within_class


def compute_betweenclass(class_mean):
    dist = np.subtract(class_mean[0], class_mean[1])
    dist = dist.reshape(class_mean.shape[1], 1)
    between_class = np.matmul(dist, dist.T)
    print("Between-class scatter matrix SB: {}".format(between_class))
    return between_class


def compute_eigen(A):
    eigenvalues, eigenvectors = np.linalg.eig(A)
    idx = eigenvalues.argsort()[::-1]                          # sort largest
    weight = eigenvectors[:, idx][:, :1]
    print("Fisher’s linear discriminant: {}".format(weight))
    return weight


def knn(x_train, y_train, x_test):
    y_pred = []
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    for test_idx in range(test_size):
        all_dist = np.zeros(train_size)
        for train_idx in range(train_size):
            partial_dist = (x_test[test_idx] - x_train[train_idx]) ** 2
            dist = np.sqrt(np.sum(partial_dist))
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


def plot_result(x_train, y_train, weight, class_mean):
    fig = plt.figure()
    plt.xlim(-2, 5)
    plt.ylim(-2, 5)
    # Plot project line
    project_x = np.linspace(-5, 5, 30)
    project_slope = weight[1] / weight[0]
    # Through origin, hence b = 0
    project_intercept = 0
    project_func = np.poly1d([project_slope, project_intercept])
    project_y = project_func(project_x)
    plt.plot(project_x, project_y, c='red')
    plt.title("Projection line: w = {}, b = {}"
              .format(project_slope[0], project_intercept))
    # Plot decision boundary
    # middle_mean = np.mean(class_mean, axis=0)
    # Since orthogonal to project line
    decision_slope = (-1) / project_slope
    # decision_intercept = middle_mean[1] - decision_slope * middle_mean[0]
    # decision_func = np.poly1d([decision_slope[0], decision_intercept[0]])
    # decision_x = np.linspace(-5, 5, 30)
    # decision_y = decision_func(decision_x)
    # plt.plot(decision_x, decision_y, c='g')
    # Plot training data
    colors = ['yellow', 'magenta']
    for class_idx in range(CLASS):
        match_idx = np.where(y_train == class_idx)
        match_data = x_train[match_idx]
        plt.scatter(match_data[:, 0], match_data[:, 1], c=colors[class_idx])
        # Plot project line of each point
        for idx, data in enumerate(match_data):
            decision_intercept = data[1] - decision_slope * data[0]
            a = np.array([[project_slope[0], -1],
                         [decision_slope[0], -1]])
            b = np.array([-project_intercept, -decision_intercept[0]])
            lower_data = np.linalg.solve(a, b)
            plt.scatter(lower_data[0], lower_data[1], c=colors[class_idx])
            plt.plot([data[0], lower_data[0]], [data[1], lower_data[1]],
                     c='b', alpha=0.3)
    plt.savefig('visualization.png')


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = read_csv()
    class_mean = compute_mean(x_train, y_train)
    within_class = compute_withinclass(x_train, y_train, class_mean)
    between_class = compute_betweenclass(class_mean)
    Sw_inverse_Sb = np.matmul(np.linalg.pinv(within_class), between_class)
    weight = compute_eigen(Sw_inverse_Sb)
    lower_dimension_train = np.matmul(x_train, weight)
    lower_dimension_test = np.matmul(x_test, weight)
    y_pred = knn(lower_dimension_train, y_train, lower_dimension_test)
    compute_accuracy(y_pred, y_test)
    plot_result(x_train, y_train, weight, class_mean)