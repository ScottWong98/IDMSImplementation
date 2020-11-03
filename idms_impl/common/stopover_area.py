import numpy as np
from idms_impl.common.improved_dbscan import ImprovedDBSCAN
from collections import Counter


class StopoverArea:

    def __init__(self):
        self.sn = 0
        self.duration = 0
        self.d = 0.
        self.coordinate = []
        self.category = []

    def __lt__(self, other):
        return self.duration >= other.duration


class StopoverAreaMining:

    def __init__(self, tr_dict, eps, min_duration):
        self.tr_dict = tr_dict
        self.__eps = eps
        self.__min_duration = min_duration
        self.__duration_list = []
        self.__coordinate_list = []
        self.__clusters = []
        self.__cluster_core_dict = []

    def run(self):
        for user_id, user_tr_dict in self.tr_dict.items():
            # 获取当前用户所有轨迹点的duration列表和坐标列表
            self.__get_all_plist(user_tr_dict)
            # 获取每个轨迹点的聚类结果
            self.__gen_clusters()
            # 获取每个聚类的中心点坐标
            self.__get_cluster_core()
            # 根据聚类结果对每条轨迹进行更新（更新轨迹点的聚类标签，删除无效轨迹点，合并轨迹点）
            new_user_tr = {}
            for date, tr in user_tr_dict.items():
                tr_len = len(tr)
                # 更新轨迹点的聚类标签，删除无效轨迹点
                tmp_tr = self.__delete_invalid_point(tr, self.__clusters[:tr_len])
                # 合并轨迹点
                tmp_tr = self.__merge_adjacent_points(tmp_tr)

                new_user_tr[date] = tmp_tr
                self.__clusters = self.__clusters[tr_len:]
            self.tr_dict[user_id] = new_user_tr

    def __merge_adjacent_points(self, tr):
        """合并两个连续的聚类标签重复点"""
        tmp_tr = []
        tmp_p = None
        for idx, p in enumerate(tr):
            if idx == 0 or p.cluster_flag != tr[idx - 1].cluster_flag:
                if idx != 0:
                    tmp_p.coordinate = self.__cluster_core_dict[tmp_p.cluster_flag]
                    tmp_tr.append(tmp_p)
                tmp_p = p
            else:
                tmp_p.duration += p.duration
                tmp_p.total_data += p.total_data
                tmp_p.time[1] = p.time[1]
        tmp_p.coordinate = self.__cluster_core_dict[tmp_p.cluster_flag]
        tmp_tr.append(tmp_p)
        return tmp_tr

    @classmethod
    def __delete_invalid_point(cls, tr, clusters):
        """更新轨迹点的聚类标签，删除无效轨迹点"""
        for idx, p in enumerate(tr):
            p.cluster_flag = clusters[idx]

        return [p for p in tr if p.cluster_flag != 0]

    def __get_all_plist(self, user_tr_dict):
        """获取user所有轨迹点的duration列表和坐标列表"""
        self.__duration_list, self.__coordinate_list = [], []
        for date, tr in user_tr_dict.items():
            for p in tr:
                self.__duration_list.append(p.duration)
                self.__coordinate_list.append(p.coordinate)

    def __gen_clusters(self):
        """使用ImprovedDBSCAN算法进行轨迹点的聚类"""
        # 改成相应的数组格式
        coordinate_list = np.mat(self.__coordinate_list).transpose()

        # 进行聚类
        improved_dbscan = ImprovedDBSCAN(self.__duration_list, coordinate_list, self.__eps, self.__min_duration)
        self.__clusters = improved_dbscan.gen_cluster()

    def __get_cluster_core(self):
        self.__cluster_core_dict = {}
        for idx, cf in enumerate(self.__clusters):
            if cf not in self.__cluster_core_dict:
                self.__cluster_core_dict[cf] = []
            self.__cluster_core_dict[cf].append(self.__coordinate_list[idx])
        for cf, xy_list in self.__cluster_core_dict.items():
            self.__cluster_core_dict[cf] = self.__get_core_coordinate(xy_list)

    @classmethod
    def __get_core_coordinate(cls, xy_list):
        """获取停留区域中心点"""
        _x, _y = 0.0, 0.0
        for xy in xy_list:
            _x += xy[0]
            _y += xy[1]
        _len = len(xy_list)
        return _x / _len, _y / _len


class SemanticTagConversion:

    def __init__(self, tr_dict, n, theta, knn_model, poi_list):
        self.tr_dict = tr_dict
        self.__n = n
        self.__theta = theta
        self.__knn_model = knn_model
        self.__poi_list = poi_list
        self.__area_dict = {}
        self.__sum_duration = 0
        self.__sum_d = 0.

    def run(self):
        for user_id, user_tr_dict in self.tr_dict.items():
            # 获取主要驻留区域
            sr_dict = self.__get_main_area(user_tr_dict)
            # 利用KNN找到距离各点最近的POI信息
            semantic_dict = {}
            for cluster_id, area in self.__area_dict.items():
                result = self.__knn_model.kneighbors([area.coordinate])
                category_list = [self.__poi_list[index].category for index in result[1][0]]
                category_list = np.array(category_list)
                # print(category_list[:, 0])
                category_dict = Counter(category_list[:, 0])

                for b_ctg in category_dict:
                    if self.__judge_main_area(sr_dict, cluster_id, b_ctg):
                        semantic_dict[cluster_id] = self.__get_multi_category(b_ctg, category_list)
                        break
                # 说明对于家和工作这种地点，在k个邻居里面没有找到匹配的
                # 暂时设置为K个邻居中出现次数最多的那个
                if cluster_id not in semantic_dict:
                    b_ctg = list(category_dict.items())[0][0]
                    semantic_dict[cluster_id] = self.__get_multi_category(b_ctg, category_list)
            for date, tr in user_tr_dict.items():
                for point in tr:
                    point.sr = semantic_dict[point.cluster_flag]

    def __get_main_area(self, user_tr_dict):
        """获取主要驻留区域"""
        sr_dict, sr_list = {}, []
        # 处理各停留区的数据信息
        self.__handle_area(user_tr_dict)
        # 对各停留区依据duration进行降序排列
        self.__sort_area_in_des_order()
        # 提取dura累计占比在 theta 前的sr，存入sr_list
        tmp_sum_duration = 0.0
        for cluster_id, area in self.__area_dict.items():
            tmp_sum_duration += area.duration
            if tmp_sum_duration / self.__sum_duration <= self.__theta:
                sr_list.append(cluster_id)
        n_sr = len(sr_list)
        if n_sr == 1:
            sr_dict[sr_list[0]] = "HOME"
        elif n_sr == 2:
            idx1, idx2 = sr_list[0], sr_list[1]
            home_probe1 = self.__home_probe(self.__area_dict[idx1].sn, self.__area_dict[idx1].d)
            home_probe2 = self.__home_probe(self.__area_dict[idx2].sn, self.__area_dict[idx2].d)
            if home_probe1 > home_probe2:
                sr_dict[idx1], sr_dict[idx2] = "HOME", "WORK"
            else:
                sr_dict[idx1], sr_dict[idx2] = "WORK", "HOME"
        elif n_sr == 3:
            idx1, idx2, idx3 = sr_list[0], sr_list[1], sr_list[2]
            d1, d2, d3 = self.__area_dict[idx1].d, self.__area_dict[idx2].d, self.__area_dict[idx3].d
            if d1 > (d2 + d3) / 2:
                sr_dict[idx1], sr_dict[idx2], sr_dict[idx3] = "WORK", "HOME", "HOME"
            else:
                sr_dict[idx1], sr_dict[idx2], sr_dict[idx3] = "HOME", "WORK", "WORK"

        return sr_dict

    def __handle_area(self, user_tr_dict):
        """计算停留区域的sn，duration，d"""
        for date, tr in user_tr_dict.items():
            cluster_set = set()
            for p in tr:
                cf = p.cluster_flag
                cluster_set.add(cf)
                if cf not in self.__area_dict:
                    self.__area_dict[cf] = StopoverArea()
                self.__area_dict[cf].duration += p.duration
                self.__area_dict[cf].d += p.total_data
                self.__area_dict[cf].coordinate = p.coordinate

            for cf in cluster_set:
                self.__area_dict[cf].sn += 1
        for cf, area in self.__area_dict.items():
            area.d /= self.__n
            self.__sum_duration += area.duration
            self.__sum_d += area.d

    def __sort_area_in_des_order(self):
        """根据停留区域的duration进行降序排列"""
        result = sorted(self.__area_dict.items(), key=lambda kv: (kv[1], kv[0]))
        # print(result)
        self.__area_dict = dict(result)

    def __home_probe(self, sn, d):
        sn_probe = sn / self.__n
        if self.__sum_d != 0:
            d_probe = d / self.__sum_d
        else:
            d_probe = 1
        return sn_probe + (1 - d_probe)

    @classmethod
    def __judge_main_area(cls, sr_dict, cluster_id, base_category):
        """判断是否满足主要驻留区域的条件"""
        if cluster_id not in sr_dict:
            return True
        area_flag = sr_dict[cluster_id]
        # TODO: 怎么判断是否为家
        # home_list = ['商务住宅', '住宿服务', '生活服务', '地名地址信息']
        # TODO: 怎么判断是否为公司
        # work_list = ['公司企业']
        home_list = []
        work_list = []

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
    def __get_multi_category(cls, base_category, category_list):
        for c in category_list:
            if base_category == c[0]:
                return c.tolist()

