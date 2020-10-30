import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KDTree


class ImprovedDBSCAN:
    """扩展的DBSCAN算法"""
    UNCLASSIFIED = False
    NOISE = 0

    def __init__(self, raw_data, loc_data, eps, min_duration):
        self.raw_data = raw_data
        self.loc_data = loc_data
        self.eps = eps
        self.min_duration = min_duration

    def region_query_kd(self, point_id):
        t_loc_data = self.loc_data.transpose()
        tree = KDTree(t_loc_data)
        return tree.query_radius(t_loc_data[point_id], r=self.eps)[0].tolist()

    def duration_sum_calculation(self, seeds):
        duration_sum = 0
        for i in seeds:
            # TODO: FIX: may be not 4
            duration_sum += self.raw_data[i][4]
        return duration_sum

    def expand_cluster(self, cluster_list, point_id, cluster_id):
        seeds = self.region_query_kd(point_id)
        if self.duration_sum_calculation(seeds) < self.min_duration:
            cluster_list[point_id] = self.NOISE
            return False
        else:
            cluster_list[point_id] = cluster_id
            for seed_id in seeds:
                cluster_list[seed_id] = cluster_id
            while len(seeds) > 0:
                current_point = seeds[0]
                query_result = self.region_query_kd(current_point)
                if self.duration_sum_calculation(query_result) >= self.min_duration:
                    for i in range(len(query_result)):
                        result_point = query_result[i]
                        if cluster_list[result_point] == self.UNCLASSIFIED:
                            cluster_list[result_point] = cluster_id
                            seeds.append(result_point)
                        elif cluster_list[result_point] == self.NOISE:
                            cluster_list[result_point] = cluster_id
                seeds = seeds[1:]
            return True

    def generate_cluster(self):
        cluster_id = 1
        points_num = self.loc_data.shape[1]
        cluster_list = [self.UNCLASSIFIED] * points_num

        for point_id in range(points_num):
            if cluster_list[point_id] == self.UNCLASSIFIED:
                if self.expand_cluster(cluster_list, point_id, cluster_id):
                    cluster_id = cluster_id + 1
        return cluster_list, cluster_id - 1

    def plot_feature(self, labels, cluster_num):
        labels = np.array(labels)
        core_sample_mask = np.zeros_like(labels, dtype=bool)
        core_sample_indices = []
        for idx, v in enumerate(labels):
            if v != 0:
                core_sample_indices.append(idx)
        core_sample_mask[core_sample_indices] = True

        unique_labels = set(labels)

        colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]

        plt.figure(figsize=(10,6))
        #plt.xticks()

        for k, col in zip(unique_labels, colors):
            if k == 0:
                col = [0, 0, 0, 1]
            class_member_mask = (labels == k)
            X = np.array(self.loc_data.transpose())

            xy = X[class_member_mask & core_sample_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),markeredgecolor='k', markersize=6)

            xy = X[class_member_mask & ~core_sample_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=3)
        plt.show()

