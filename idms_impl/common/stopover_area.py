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


class StopoverAreaSet:

    def __init__(self, tr_dict):
        self.tr_dict = tr_dict
        pass

    def stopover_area_mining(self, eps, min_duration):
        # 遍历用户
        for user_id, user_rt_dict in self.tr_dict.items():
            # 对该用户的所有轨迹进行更新，返回停留区域轨迹
            user_tr = UserTr(user_rt_dict)
            self.tr_dict[user_id] = user_tr.update_user_tr_dict(eps, min_duration)

    def semantic_tag_conversion(self, n, theta, knn_model, poi_list):
        for user_id, user_rt_dict in self.tr_dict.items():
            user_tr = UserTr(user_rt_dict)
            user_tr.semantic_tag_conversion(n, theta, knn_model, poi_list)
            self.tr_dict[user_id] = user_tr.user_tr_dict


class UserTr:
    """对某一用户的所有轨迹进行停留区域挖掘以及语义的转化"""
    def __init__(self, user_tr_dict):
        self.user_tr_dict = user_tr_dict
        self.area_dict = {}
        self.cluster_core_dict = {}
        self.n = None
        self.theta = None
        self.sum_d = 0.0
        self.sum_duration = 0

    def update_user_tr_dict(self, eps, min_duration):
        # 获取当前用户所有轨迹点的duration列表和坐标列表
        duration_list, coordinate_list = self.__get_all_plist(self.user_tr_dict)
        # 获取每个轨迹点的聚类结果
        clusters = self.__gen_clusters(duration_list, coordinate_list, eps, min_duration)
        # 获取每个聚类的中心点坐标
        self.__get_cluster_core(clusters, coordinate_list)
        # 根据聚类结果对每条轨迹进行更新（更新轨迹点的聚类标签，删除无效轨迹点，合并轨迹点）
        new_user_tr = {}
        for date, tr in self.user_tr_dict.items():
            tr_len = len(tr)
            # 更新轨迹点的聚类标签，删除无效轨迹点
            tmp_tr = self.__delete_invalid_point(tr, clusters[:tr_len])
            # 合并轨迹点
            tmp_tr = self.__merge_adjacent_points(tmp_tr)

            new_user_tr[date] = tmp_tr
            clusters = clusters[tr_len:]

        return new_user_tr

    def semantic_tag_conversion(self, n, theta, knn_model, poi_list):
        self.n = n
        self.theta = theta
        # 获取主要驻留区域
        sr_dict = self.__get_main_area()
        # 利用KNN找到距离各点最近的POI信息
        semantic_dict = {}
        for cluster_id, area in self.area_dict.items():
            result = knn_model.kneighbors([area.coordinate])
            category_list = [poi_list[index].category for index in result[1][0]]
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
        for date, tr in self.user_tr_dict.items():
            for point in tr:
                point.sr = semantic_dict[point.cluster_flag]

    @classmethod
    def __get_multi_category(cls, base_category, category_list):
        for c in category_list:
            if base_category == c[0]:
                return c.tolist()

    @classmethod
    def __judge_main_area(cls, sr_dict, cluster_id, base_category):
        """判断是否满足主要驻留区域的条件"""
        if cluster_id not in sr_dict:
            return True
        area_flag = sr_dict[cluster_id]
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

    def __get_main_area(self):
        """获取主要驻留区域"""
        sr_dict, sr_list = {}, []
        # 处理各停留区的数据信息
        self.__handle_area()
        # 对各停留区依据duration进行降序排列
        self.__sort_area_in_des_order()
        # 提取dura累计占比在 theta 前的sr，存入sr_list
        tmp_sum_duration = 0.0
        for cluster_id, area in self.area_dict.items():
            tmp_sum_duration += area.duration
            if tmp_sum_duration / self.sum_duration <= self.theta:
                sr_list.append(cluster_id)
        n_sr = len(sr_list)
        if n_sr == 1:
            sr_dict[sr_list[0]] = "HOME"
        elif n_sr == 2:
            idx1, idx2 = sr_list[0], sr_list[1]
            home_probe1 = self.__home_probe(self.area_dict[idx1].sn, self.area_dict[idx1].d)
            home_probe2 = self.__home_probe(self.area_dict[idx2].sn, self.area_dict[idx2].d)
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

    def __sort_area_in_des_order(self):
        """根据停留区域的duration进行降序排列"""
        result = sorted(self.area_dict.items(), key=lambda kv: (kv[1], kv[0]))
        # print(result)
        self.area_dict = dict(result)

    def __home_probe(self, sn, d):
        sn_probe = sn / self.n
        if self.sum_d != 0:
            d_probe = d / self.sum_d
        else:
            d_probe = 1
        return sn_probe + (1 - d_probe)

    def __handle_area(self):
        """计算停留区域的sn，duration，d"""
        for date, tr in self.user_tr_dict.items():
            cluster_set = set()
            for p in tr:
                cf = p.cluster_flag
                cluster_set.add(cf)
                if cf not in self.area_dict:
                    self.area_dict[cf] = StopoverArea()
                self.area_dict[cf].duration += p.duration
                self.area_dict[cf].d += p.total_data
                self.area_dict[cf].coordinate = p.coordinate

            for cf in cluster_set:
                self.area_dict[cf].sn += 1
        for cf, area in self.area_dict.items():
            area.d /= self.n
            self.sum_duration += area.duration
            self.sum_d += area.d

    @classmethod
    def __delete_invalid_point(cls, tr, clusters):
        """更新轨迹点的聚类标签，删除无效轨迹点"""
        for idx, p in enumerate(tr):
            p.cluster_flag = clusters[idx]

        return [p for p in tr if p.cluster_flag != 0]

    def __merge_adjacent_points(self, tr):
        """合并两个连续的聚类标签重复点"""
        tmp_tr = []
        tmp_p = None
        for idx, p in enumerate(tr):
            if idx == 0 or p.cluster_flag != tr[idx - 1].cluster_flag:
                if idx != 0:
                    tmp_p.coordinate = self.cluster_core_dict[tmp_p.cluster_flag]
                    tmp_tr.append(tmp_p)
                tmp_p = p
            else:
                tmp_p.duration += p.duration
                tmp_p.total_data += p.total_data
                tmp_p.time[1] = p.time[1]
        tmp_p.coordinate = self.cluster_core_dict[tmp_p.cluster_flag]
        tmp_tr.append(tmp_p)
        return tmp_tr

    def __get_cluster_core(self, clusters, coordinate_list):
        for idx, cf in enumerate(clusters):
            if cf not in self.cluster_core_dict:
                self.cluster_core_dict[cf] = []
            self.cluster_core_dict[cf].append(coordinate_list[idx])
        for cf, xy_list in self.cluster_core_dict.items():
            self.cluster_core_dict[cf] = self.__get_core_coordinate(xy_list)

    @classmethod
    def __get_core_coordinate(cls, xy_list):
        """获取停留区域中心点"""
        _x, _y = 0.0, 0.0
        for xy in xy_list:
            _x += xy[0]
            _y += xy[1]
        _len = len(xy_list)
        return _x / _len, _y / _len

    @classmethod
    def __gen_clusters(cls, duration_list, coordinate_list, eps, duration):
        """使用ImprovedDBSCAN算法进行轨迹点的聚类"""
        # 改成相应的数组格式
        coordinate_list = np.mat(coordinate_list).transpose()

        # 进行聚类
        improved_dbscan = ImprovedDBSCAN(duration_list, coordinate_list, eps, duration)
        clusters = improved_dbscan.gen_cluster()

        return clusters

    @classmethod
    def __get_all_plist(cls, user):
        """获取user所有轨迹点的duration列表和坐标列表"""
        duration_list, coordinate_list = [], []
        for date, tr in user.items():
            for p in tr:
                duration_list.append(p.duration)
                coordinate_list.append(p.coordinate)

        return duration_list, coordinate_list
