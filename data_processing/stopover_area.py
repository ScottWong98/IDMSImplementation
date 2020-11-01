import numpy as np
from collections import Counter


class StopoverArea:

    def __init__(self):
        self.sn = 0
        self.duration = 0
        self.d = 0.0
        # 停留区中心区域坐标
        self.coordinate = []
        # 停留区对应的POI类别
        self.category = []

    def __lt__(self, other):
        return self.duration >= other.duration


class StopoverAreaSet:

    def __init__(self, tr_dict, n, theta, knn_model, poi_list):
        # 用户所有轨迹集合，一个字典，可根据时间获取到当天的轨迹
        self.tr_dict = tr_dict
        # 停留区域字典，key为停留区域聚类标签，value为StopoverArea
        self.area_dict = {}
        self.n = n
        self.theta = theta
        self.sum_d = 0.0
        self.sum_duration = 0
        self.sr_dict = {}
        self.knn_model = knn_model
        self.poi_list = poi_list

    def get_semantic_dict(self):
        # 1. 获取主要驻留区域
        self.sr_dict = self.get_main_stopover_area()

        # 2. 从之前训练的KNN模型中找到距离个点最近的POI信息
        neigh = self.knn_model
        semantic_dict = {}
        for (cluster_id, area) in self.area_dict.items():
            result = neigh.kneighbors([area.coordinate])
            category_list = [self.poi_list[index].category for index in result[1][0]]
            category_list = np.array(category_list)
            # print(category_list[:, 0])
            category_dict = Counter(category_list[:, 0])

            for b_ctg in category_dict:
                if self.judge_main_area(cluster_id, b_ctg):
                    semantic_dict[cluster_id] = self.get_multi_category(b_ctg, category_list)
                    break
            # 说明对于家和工作这种地点，在k个邻居里面没有找到匹配的
            # 暂时设置为K个邻居中出现次数最多的那个
            if cluster_id not in semantic_dict:
                b_ctg = list(category_dict.items())[0][0]
                semantic_dict[cluster_id] = self.get_multi_category(b_ctg, category_list)
        return semantic_dict

    def get_main_stopover_area(self):
        sr_dict = {}
        sr_list = []

        # 1. 处理各停留区的数据信息
        self.handle_all_area()
        # self.output()

        # 2. 对各停留区依据duration进行降序排列
        self.sort_in_des_order()
        # self.output()

        # 3. 提取dura累计占比在 theta 前的sr，存入sr_list
        tmp_sum_duration = 0.0
        for (cluster_id, area) in self.area_dict.items():
            tmp_sum_duration += area.duration
            # print(area.duration, tmp_sum_duration)
            # print(tmp_sum_duration / self.sum_duration)
            if tmp_sum_duration / self.sum_duration <= self.theta:
                sr_list.append(cluster_id)
        n_sr = len(sr_list)
        if n_sr == 1:
            sr_dict[sr_list[0]] = "HOME"
        elif n_sr == 2:
            idx1, idx2 = sr_list[0], sr_list[1]
            home_probe1 = self.home_probe(self.area_dict[idx1].sn, self.area_dict[idx1].d)
            home_probe2 = self.home_probe(self.area_dict[idx2].sn, self.area_dict[idx2].d)
            if home_probe1 > home_probe2:
                sr_dict[idx1], sr_dict[idx2] = "HOME", "WORK"
            else:
                sr_dict[idx1], sr_dict[idx2] = "WORK", "HOME"
        elif n_sr == 3:
            idx1, idx2, idx3 = sr_list[0], sr_list[1], sr_list[2]
            d1, d2, d3 = self.area_dict[idx1].d, self.area_dict[idx2].d, self.area_dict[idx3].d
            if d1 > (d2 + d3) / 2:
                sr_dict[idx1], sr_dict[idx2], sr_dict[idx3] = "WORK", "HOME", "HOME"
            else:
                sr_dict[idx1], sr_dict[idx2], sr_dict[idx3] = "HOME", "WORK", "WORK"

        return sr_dict

    def home_probe(self, sn, d):
        sn_probe = sn / self.n
        if self.sum_d != 0:
            d_probe = d / self.sum_d
        else:
            d_probe = 1
        return sn_probe + (1 - d_probe)

    def handle_all_area(self):
        """计算各停留区内的sn，duration，d"""
        for (date, tr) in self.tr_dict.items():
            cluster_set = set()
            for point in tr.plist:
                cluster_set.add(point.sr)
                if point.sr not in self.area_dict:
                    self.area_dict[point.sr] = StopoverArea()
                self.area_dict[point.sr].duration += point.duration
                self.area_dict[point.sr].d += point.total_data
                self.area_dict[point.sr].coordinate = [point.longitude, point.latitude]

            for cluster_id in cluster_set:
                self.area_dict[cluster_id].sn += 1
        for (cluster_id, area) in self.area_dict.items():
            area.d /= self.n
            self.sum_duration += area.duration
            self.sum_d += area.d

    def sort_in_des_order(self):
        """根据停留区域的duration进行降序排列"""
        result = sorted(self.area_dict.items(), key=lambda kv: (kv[1], kv[0]))
        # print(result)
        self.area_dict = dict(result)

    def judge_main_area(self, cluster_id, base_category):
        if cluster_id not in self.sr_dict:
            return True
        area_flag = self.sr_dict[cluster_id]
        # TODO: 怎么判断是否为家
        home_list = ['商务住宅', '住宿服务', '生活服务', '地名地址信息']
        # TODO: 怎么判断是否为公司
        work_list = ['公司企业']

        if area_flag == 'HOME':
            match_num = [i for i in home_list if base_category == i]
            if len(match_num) == 0:
                return False
            else:
                return True
        else:
            match_num = [i for i in work_list if base_category == i]
            if len(match_num) == 0:
                return False
            else:
                return True

    @classmethod
    def get_multi_category(cls, base_category, category_list):
        for c in category_list:
            if base_category == c[0]:
                return c.tolist()

    def output(self):
        print('=' * 40)
        print("Stopover Area Information")
        print('=' * 40)
        for (idx, area) in self.area_dict.items():
            print(idx)
            print('=' * 20)
            print("sn: %s duration: %s d: %s" % (area.sn, area.duration, area.d))
