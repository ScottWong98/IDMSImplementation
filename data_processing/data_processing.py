import csv
import datetime
import time
import numpy as np
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


class ClusterProcess:

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
        # clusters_mat = np.mat(clusters).transpose()
        # fig = plt.figure()
        # scatter_colors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
        # ax = fig.add_subplot(111)
        # for i in range(cluster_num + 1):
        #     color_style = scatter_colors[i % len(scatter_colors)]
        #     sub_cluster = self.loc_data[:, np.nonzero(clusters_mat[:, 0].A == i)]
        #     ax.scatter(sub_cluster[0, :].flatten().A[0], sub_cluster[1, :].flatten().A[0], c=color_style, s=50)
        # plt.show()


# class UserTrajectory:
#
#     def __init__(self, raw_data, loc_data, eps, min_duration):
#         self.raw_data = raw_data
#         self.loc_data = loc_data
#         self.eps = eps
#         self.min_duration = min_duration
#
#     def get_SRtr(self):
#
#
#     def get_total_days(self):
#
#     def get_area_total_days(self):


class DataLoad:

    def __init__(self, filename):
        self.filename = filename

    @classmethod
    def string2datetime(cls, str_datetime):
        return datetime.datetime.strptime(str_datetime, "%Y%m%d%H%M%S")

    def load_data(self):
        data = {}
        with open(self.filename, 'r', encoding='UTF-8') as f:
            f_csv = csv.DictReader(f)
            items = list(f_csv)
            for item in items:
                msisdn = item['MSISDN']
                time_in = self.string2datetime(item['STIME'])
                time_out = self.string2datetime(item['END_TIME'])
                longitude = float(item['NUMBERITUDE'])
                latitude = float(item['LATITUDE'])
                duration = int(item['DURATION'])
                day_number = int(item['DAY_NUMBER'])

                dt = item['DATA_TOTAL']
                total_data = 0.0
                if dt != '':
                    total_data = float(dt)

                _list = [time_in, time_out, longitude, latitude, duration, day_number, total_data]
                if msisdn in data:
                    data[msisdn].append(_list)
                else:
                    data[msisdn] = [_list]
        return data


def string2datetime(_raw_datetime):
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


def tmp_extract_loc_data(data):
    loc_data = []
    for item in data:
        loc_data.append(item[2:4])
    return loc_data


def main():
    data_load = DataLoad('../resource/test_input_2.csv')
    user_data = data_load.load_data()

    for user_id in user_data:
        raw_data = user_data[user_id]
        loc_data = tmp_extract_loc_data(raw_data)
        loc_data = np.mat(loc_data).transpose()
        cluster_process = ClusterProcess(raw_data, loc_data, eps=0.01, min_duration=5000)
        clusters, cluster_num = cluster_process.generate_cluster()
        print("Cluster num= ", cluster_num)
        # cluster_process.plot_feature(clusters, cluster_num)


if __name__ == '__main__':
    start = time.process_time()
    main()
    end = time.process_time()
    print("Finish all in %s " % str(end - start))

