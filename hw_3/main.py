import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def calculate_distance_matrix(n):
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j] = matrix[j][i] = round(random.uniform(10, 99), 1)
    return [np.asmatrix(np.array(matrix)), matrix]


def prim_min_ost_tree(mtx):
    val_more_than_max = 100
    min_mtx = [[0 for _ in range(n)] for _ in range(n)]
    circles_indexes = [0]

    while len(circles_indexes) < len(mtx):
        min_index_from = 0
        min_index_to = 0
        min_score = val_more_than_max
        for i in circles_indexes:
            for j in range(len(mtx)):
                if j in circles_indexes:
                    continue
                if mtx[i][j] != 0 and mtx[i][j] <= min_score:
                    min_score = mtx[i][j]
                    min_index_to = j
                    min_index_from = i
        circles_indexes.append(min_index_to)
        min_mtx[min_index_from][min_index_to] = min_score

    return [np.asmatrix(np.array(min_mtx)), min_mtx]


class Line:
    def __init__(self, size, circle_from, circle_to):
        self.size = size
        self.circle_from = circle_from
        self.circle_to = circle_to


def clusterize(mtx, clusters_count):
    new_mtx = mtx
    largest_lines = []
    for i in range(len(mtx)):
        for j in range(len(mtx)):
            if mtx[i][j] != 0:
                line = Line(mtx[i][j], i, j)
                largest_lines.append(line)
    largest_lines.sort(key=lambda x: x.size, reverse=True)
    for i in range(clusters_count - 1):
        new_mtx[largest_lines[i].circle_from][largest_lines[i].circle_to] = 0
    return np.asmatrix(np.array(new_mtx))


if __name__ == '__main__':
    n = 10
    matrixes = calculate_distance_matrix(n)
    G = nx.from_numpy_matrix(matrixes[0])
    posG = nx.spring_layout(G, k=0.9 / np.sqrt(len(G.nodes())))
    nx.draw_networkx(G, posG)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, posG, edge_labels=labels)
    plt.show()

    min_mtxes = prim_min_ost_tree(matrixes[1])
    G1 = nx.from_numpy_matrix(min_mtxes[0])
    pos = nx.spring_layout(G1, k=0.9 / np.sqrt(len(G1.nodes())))
    nx.draw_networkx(G1, pos)
    labels = nx.get_edge_attributes(G1, 'weight')
    nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels)
    plt.show()

    clusters = clusterize(min_mtxes[1], 5)
    G2 = nx.from_numpy_matrix(clusters)
    pos2 = nx.spring_layout(G2, k=0.9 / np.sqrt(len(G2.nodes())))
    nx.draw_networkx(G2, pos2)
    labels = nx.get_edge_attributes(G2, 'weight')
    nx.draw_networkx_edge_labels(G2, pos2, edge_labels=labels)
    plt.show()
