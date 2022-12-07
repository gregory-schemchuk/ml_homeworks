import pygame
import numpy as np


def dist(pntA, pntB):
    return np.sqrt((pntA[0] - pntB[0]) ** 2 + (pntA[1] - pntB[1]) ** 2)


def find_neighbors(input_pnt, points, eps):
    neighbors = []
    for point in points:
        if dist(input_pnt, point) < eps and input_pnt != point:
            neighbors.append(point)
    return neighbors


def dbscan(points):
    min_pts = 3
    eps = 30
    clustered_points = set()
    clusters = []
    yellows = set()

    flag = ['red'] * len(points)
    for i in range(len(points)):
        number_pts = 0
        for j in range(len(points)):
            if dist(points[i], points[j]) < eps and i != j:
                number_pts += 1
        if number_pts >= min_pts:
            flag[i] = 'green'
    for i in range(len(points)):
        if flag[i] == 'red':
            for j in range(len(points)):
                if i != j and flag[j] == 'green' and dist(points[i], points[j]) < eps:
                    flag[i] = 'yellow'
                    yellows.add(points[i])

    for i in range(len(points)):
        if flag[i] != 'green' or points[i] in clustered_points:
            continue
        neighbors = find_neighbors(points[i], points, eps)
        cluster = [points[i]]
        clustered_points.add(points[i])
        while neighbors:
            n = neighbors.pop()
            if n in yellows:
                continue
            if n not in cluster:
                n_neighbors = find_neighbors(n, points, eps)
                neighbors.extend(n_neighbors)
                cluster.append(n)
                clustered_points.add(n)
        clusters.append(cluster)

    for yellow in yellows:
        nearest_neighbors = find_neighbors(yellow, points, eps)
        i = 0
        while nearest_neighbors[i] not in clustered_points:
            i += 1
        for cluster in clusters:
            if nearest_neighbors[i] in cluster:
                cluster.append(yellow)
                break

    return flag, clusters


def paint():
    points = []
    flags, clusters = [], []
    R = 7
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode([800, 600])
    screen.fill(color='white')
    pygame.display.update()
    colors = ['blue', 'blueviolet', 'burlywood1', 'cadetblue1', 'chartreuse2', 'coral1', 'cyan2', 'deeppink']

    play = True

    while play:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                play = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    points.append((event.pos[0], event.pos[1]))
                    pygame.draw.circle(screen, color='black', center=event.pos, radius=R)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN:
                    flags, clusters = dbscan(points)
                    for i in range(len(points)):
                        pygame.draw.circle(screen, color=flags[i], center=points[i], radius=R)

                if event.key == pygame.K_SPACE:
                    for i in range(len(clusters)):
                        for j in range(len(clusters[i])):
                            pygame.draw.circle(screen, color=colors[i], center=clusters[i][j], radius=R)

        pygame.display.update()
        clock.tick(60)


if __name__ == '__main__':
    paint()
