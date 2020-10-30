import numpy as np
from data_processing.improved_dbscan import ImprovedDBSCAN
from data_processing.stopover_area import StopoverAreaSet


class User:

    def __init__(self, user_id):
        self.user_id = user_id
        self.tr_dict = {}

    def stopover_area_excavation(self, eps, min_duration):
        # 1. 将tr_dict 中的所有轨迹点转化成一个列表
        plist, loc_plist = self.get_all_plist()

        # 2. 进行聚类
        loc_plist = np.mat(loc_plist).transpose()
        improved_dbscan = ImprovedDBSCAN(plist, loc_plist, eps, min_duration)
        clusters, cluster_num = improved_dbscan.generate_cluster()

        # 3. 更改每个轨迹点的sr值，赋予聚类标签
        self.generate_sr_tr(clusters)
        # self.output()

    def semantic_tag_conversion(self, n, theta):
        # 1. 获取语义词典（ <聚类标签：POI> ）
        stopover_area_set = StopoverAreaSet(self.tr_dict, n, theta)
        semantic_dict = stopover_area_set.get_semantic_dict()

        # print(semantic_dict)
        # 2. 将所有轨迹点的sr改为相应的POI
        self.generate_ms_tr(semantic_dict)

    def generate_sr_tr(self, clusters):
        """
        将 tr_dict 中的每一天的轨迹记录中的sr改成聚类标签
        并将对轨迹中的点进行合并、删除操作
        :return:
        """
        for (date, tr) in self.tr_dict.items():
            tr_length = len(tr.plist)
            tr.convert2sr_tr(clusters[:tr_length])
            clusters = clusters[tr_length:]
            # tr.output()

    # TODO
    def generate_ms_tr(self, semantic_dict):
        """将用户中的所有轨迹点中的sr根据semantic_dict转化成POI"""
        for (date, tr) in self.tr_dict.items():
            for point in tr.plist:
                point.sr = semantic_dict[point.sr]

    def get_all_plist(self):
        """获取该用户的所有轨迹点列表"""
        plist, loc_plist = [], []
        for (date, tr) in self.tr_dict.items():
            for idx, point in enumerate(tr.plist):
                _list = [point.time_in, point.time_out, point.longitude, point.latitude,
                         point.duration, point.day_number, point.total_data]
                plist.append(_list)
                loc_plist.append([point.longitude, point.latitude])
        return plist, loc_plist

    def output(self):
        print('=' * 40)
        print("User ID:", self.user_id)
        for (date, tr) in self.tr_dict.items():
            print('-' * 30)
            print("Date:", date)
            for idx, point in enumerate(tr.plist):
                print(point)


