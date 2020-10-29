import csv
import datetime
import time
import random
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


UNCLASSIFIED = False
NOISE = 0


def string2datetime(_raw_datetime):
    """ Convert String to Datetime """
    return datetime.datetime.strptime(_raw_datetime, "%Y%m%d%H%M%S")


def load_data_set(filename):
    data = []
    data_loc = []
    with open(filename, 'r', encoding='UTF-8') as f:
        f_csv = csv.DictReader(f)
        items = list(f_csv)

        for item in items:
            time_in = string2datetime(item['STIME'])
            time_out = string2datetime(item['END_TIME'])
            longitude = float(item['NUMBERITUDE'])
            latitude = float(item['LATITUDE'])
            duration = int(item['DURATION'])
            data.append([time_in,
                         time_out,
                         longitude,
                         latitude,
                         duration])
            data_loc.append([longitude, latitude])
    return data, data_loc


def region_query_kd(data_loc, point_id, eps):
    data_loc = data_loc.transpose()
    tree = KDTree(data_loc)
    seeds = tree.query_radius(data_loc[point_id], r=eps)[0].tolist()
    return seeds


def duration_sum_calculation(data, seeds):
    duration_sum = 0
    for i in seeds:
        duration_sum += data[i][4]
    return duration_sum


def expand_cluster(data, data_loc, cluster_list, point_id, cluster_id, eps, min_duration):
    seeds = region_query_kd(data_loc, point_id, eps)
    if duration_sum_calculation(data, seeds) < min_duration:
        cluster_list[point_id] = NOISE
        return False
    else:
        cluster_list[point_id] = cluster_id
        for seed_id in seeds:
            cluster_list[seed_id] = cluster_id
        while len(seeds) > 0:
            current_point = seeds[0]
            query_result = region_query_kd(data_loc, current_point, eps)
            if duration_sum_calculation(data, query_result) >= min_duration:
                for i in range(len(query_result)):
                    result_point = query_result[i]
                    if cluster_list[result_point] == UNCLASSIFIED:
                        cluster_list[result_point] = cluster_id
                        seeds.append(result_point)
                    elif cluster_list[result_point] == NOISE:
                        cluster_list[result_point] = cluster_id
            seeds = seeds[1:]
        return True

def generate_cluster(data, data_loc, eps, min_duration):
    cluster_id = 1
    points_num = data_loc.shape[1]
    cluster_list = [UNCLASSIFIED] * points_num

    for point_id in range(points_num):
        if cluster_list[point_id] == UNCLASSIFIED:
            if expand_cluster(data, data_loc, cluster_list, point_id, cluster_id, eps, min_duration):
                cluster_id = cluster_id + 1
    return cluster_list, cluster_id - 1


def plot_feature(data, clusters, cluster_num):
    clusters_mat = np.mat(clusters).transpose()
    fig = plt.figure()
    scatter_colors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    ax = fig.add_subplot(111)
    for i in range(cluster_num + 1):
        color_style = scatter_colors[i % len(scatter_colors)]
        sub_cluster = data[:, np.nonzero(clusters_mat[:, 0].A == i)]
        ax.scatter(sub_cluster[0, :].flatten().A[0], sub_cluster[1, :].flatten().A[0], c=color_style, s=50)
    plt.show()


def main():
    data_set, data_set_loc = load_data_set("../resource/test_input_1.csv")
    data_set_loc = np.mat(data_set_loc).transpose()
    #print(data_set_loc)
    clusters, cluster_num = generate_cluster(data_set, data_set_loc, eps=0.0005, min_duration=10000)
    print("Cluster num= ", cluster_num)
    plot_feature(data_set_loc, clusters, cluster_num)


if __name__ == '__main__':
    start = time.process_time()
    main()
    end = time.process_time()
    print("Finish all in %s " % str(end - start))