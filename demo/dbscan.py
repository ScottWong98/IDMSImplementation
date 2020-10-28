import numpy as np
import math
import time
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

UNCLASSIFIED = False
NOISE = 0


def load_data_set(filename, split_char=","):
    data_set = []
    with open(filename) as f:
        for line in f.readlines():
            curline = line.strip().split(split_char)
            fltline = list(map(float, curline))
            data_set.append(fltline)
    return data_set


def dist(a, b):
    return math.sqrt(np.power(a - b, 2).sum())


def eps_neighbor(a, b, eps):
    return dist(a, b) < eps


def region_query(data, point_id, eps):
    n_points = data.shape[1]
    seeds = []
    for i in range(n_points):
        if eps_neighbor(data[:, point_id], data[:, i], eps):
            seeds.append(i)
    return seeds


def region_query_kd(data, point_id, eps):

    data = data.transpose()
    #print(data)
    tree = KDTree(data)
    seeds = tree.query_radius(data[point_id], r=eps)[0].tolist()
    return seeds


def expand_cluster(data, cluster_result, point_id, cluster_id, eps, min_points):
    #seeds = region_query(data, point_id, eps)
    seeds = region_query_kd(data, point_id, eps)
    if len(seeds) < min_points:
        cluster_result[point_id] = NOISE
        return False
    else:
        cluster_result[point_id] = cluster_id
        for seed_id in seeds:
            cluster_result[seed_id] = cluster_id

        while len(seeds) > 0:
            current_point = seeds[0]
            #query_result = region_query(data, current_point, eps)
            query_result = region_query_kd(data, current_point, eps)
            if len(query_result) >= min_points:
                for i in range(len(query_result)):
                    result_point = query_result[i]
                    if cluster_result[result_point] == UNCLASSIFIED:
                        seeds.append(result_point)
                        cluster_result[result_point] = cluster_id
                    elif cluster_result[result_point] == NOISE:
                        cluster_result[result_point] = cluster_id
            seeds = seeds[1:]
        return True


def dbscan(data, eps, min_points):
    cluster_id = 1
    n_points = data.shape[1]
    cluster_result = [UNCLASSIFIED] * n_points

    for point_id in range(n_points):
        point = data[:, point_id]
        if cluster_result[point_id] == UNCLASSIFIED:
            if expand_cluster(data, cluster_result, point_id, cluster_id, eps, min_points):
                cluster_id = cluster_id + 1

    return cluster_result, cluster_id - 1


def plot_feature(data, clusters, cluster_num):
    mat_clusters = np.mat(clusters).transpose()
    fig = plt.figure()
    scatter_colors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(cluster_num + 1):
        color_style = scatter_colors[i % len(scatter_colors)]
        subcluster = data[:, np.nonzero(mat_clusters[:, 0].A == i)]
        ax.scatter(subcluster[0, :].flatten().A[0], subcluster[1, :].flatten().A[0], c=color_style, s=50)
    plt.show()


def main():
    dataSet = load_data_set("./788points.txt")
    print(dataSet)
    #dataSet = np.mat(dataSet).transpose()
    #clusters, cluster_num = dbscan(dataSet, 2, 15)
    #print("cluster num = ", cluster_num)
    #plot_feature(dataSet, clusters, cluster_num)


if __name__ == '__main__':
    start = time.process_time()

    main()

    end = time.process_time()
    print("Finish all in %s " % str(end - start))



    #print(dataSet)
    # print(load_data_set("./788points.txt"))