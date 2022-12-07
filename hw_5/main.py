from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay


def main():
    colors = ['blue', 'red']
    X, Y = make_blobs(n_samples=80, centers=2, cluster_std=0.6)
    clf = SVC(kernel="linear", C=1000)
    clf.fit(X, Y)
    plt.scatter(X[:, 0], X[:, 1], c=Y, s=30, cmap=ListedColormap(colors))

    DecisionBoundaryDisplay.from_estimator(clf, X, plot_method="contour", colors="k", levels=[-1, 0, 1], alpha=0.5,
                                           linestyles=["--", "-", "--"], ax=plt.gca())

    test_p = [5, 3]
    test_p_class = clf.predict([test_p])[0]
    plt.plot(test_p[0], test_p[1], c=colors[test_p_class], markersize=10, marker='X')

    plt.show()


if __name__ == '__main__':
    main()