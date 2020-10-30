import numpy as np
from data_processing.improved_dbscan import ImprovedDBSCAN


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
        self.output()

    def semantic_tag_conversion(self, n, theta):
        # 1. 获取语义词典（ <聚类标签：POI> ）
        stopover_area_set = StopoverAreaSet(self.tr_dict, n, theta)
        semantic_dict = stopover_area_set.get_semantic_dict()

        # 2. 将所有轨迹点的sr改为相应的POI



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

    def generate_ms_tr(self, semantic_dict):
        """将用户中的所有轨迹点中的sr根据semantic_dict转化成POI"""
        pass



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


class StopoverArea:

    def __init__(self):
        self.sn = 0
        self.duration = 0
        self.d = 0.0


class StopoverAreaSet:

    def __init__(self, tr_dict, n, theta):
        self.tr_dict = tr_dict
        self.area_dict = {}
        self.n = n
        self.theta = theta
        self.sum_d = 0.0
        self.sum_duration = 0

    def get_semantic_dict(self):
        sr_dict = {}
        sr_list = []
        # 1. 对各停留区依据duration进行降序排列
        self.sort()

        # 2. 提取dura累计占比在 theta 前的sr，存入sr_list
        # for
        return sr_dict

    def handle_all_area(self):
        """计算各停留区内的sn，dura，d"""
        for (date, tr) in self.tr_dict.items():
            cluster_set = set()
            for point in tr.plist:
                cluster_set.add(point.sr)
                if point.sr not in self.area_dict:
                    self.area_dict[point.sr] = StopoverArea()
                self.area_dict[point.sr].duration += point.duration
                self.area_dict[point.sr].d += point.total_data

            for cluster_id in cluster_set:
                self.area_dict[cluster_id] += 1
        for (cluster_id, area) in self.area_dict.items():
            area.d /= self.n
            self.sum_duration += area.duration
            self.sum_d += area.d

    def sort(self):
        """根据停留区域的dura进行降序排列"""
        pass
