from collections import Counter

import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def normalize_data(data):
    minmaxs = [(4.3, 7.9), (2.0, 4.4), (1.0, 6.9), (0.1, 2.5)]
    features = []
    for i, column in enumerate(data.T):
        feature_values = []
        for row in column:
            x = row
            x_normalized = (x - minmaxs[i][0]) / (minmaxs[i][1] - minmaxs[i][0])
            feature_values.append(x_normalized)
        features.append(feature_values)
    features = np.array(features).T
    return features


def find_k(features_train, features_test, classes_train, classes_test):
    accuracies = []
    for k in range(1, 41):
        classes_calculated = knn(features_train, features_test, classes_train, k)
        accuracies.append(np.sum(classes_calculated == classes_test) / len(classes_calculated))
    return accuracies.index(max(accuracies)) - 1


def knn(features_train, features_test, classes_train, k, should_print=False):
    classes_calculated = []
    for f_test in features_test:
        distances = [np.sqrt(np.sum(np.square(f_test - f_train))) for f_train in features_train]
        closest_indices = np.argsort(distances)[:k]
        class_labels = [classes_train[i] for i in closest_indices]
        classes_calculated.append(Counter(class_labels).most_common(1)[0][0])
    return classes_calculated


def plot(features, classes, lim, point=None, point_class=None):
    colors = ['red', 'green', 'blue']
    axes = {0: 'sepal length', 1: 'sepal width', 2: 'petal length', 3: 'petal width'}
    for i in range(4):
        for j in range(i + 1, 4):
            if i != j:
                plt.figure()
                plt.xlabel(axes.get(i))
                plt.ylabel(axes.get(j))
                plt.xlim(0.0, lim)
                plt.ylim(0.0, lim)
                plt.scatter(features[:, i], features[:, j], c=classes, cmap=ListedColormap(colors))
                if point is not None:
                    plt.plot(point[0][i], point[0][j], c=colors[point_class], markersize=20, marker='X')
                plt.show()


if __name__ == '__main__':
    iris = datasets.load_iris()
    data = iris.data
    normalized_data = normalize_data(iris.data)
    target = iris.target

    features_train, features_test, classes_train, classes_test = train_test_split(normalized_data, target,
                                                                                  test_size=0.3,
                                                                                  random_state=13)

    k = find_k(features_train, features_test, classes_train, classes_test)
    print(f'k: {k}')

    plot(data, target, 8.0)
    # plot(normalized_data, target, 1.0)

    new_point = np.random.rand(1, 4)
    point_class = knn(normalized_data, new_point, target, k)[0]
    plot(normalized_data, target, 1.0, new_point, point_class)
