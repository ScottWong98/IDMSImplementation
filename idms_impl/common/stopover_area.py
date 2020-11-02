import numpy as np
from idms_impl.common.improved_dbscan import ImprovedDBSCAN


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
            user_tr = UserTr(user_rt_dict, eps, min_duration)
            self.tr_dict[user_id] = user_tr.update_user_tr_dict()

    def semantic_tag_conversion(self, n, theta, knn_model):
        pass


class UserTr:
    """对某一用户的所有轨迹进行停留区域挖掘"""
    def __init__(self, user_tr_dict, eps, duration):
        self.user_tr_dict = user_tr_dict
        self.eps = eps
        self.duration = duration

    def update_user_tr_dict(self):
        # 获取当前用户所有轨迹点的duration列表和坐标列表
        duration_list, coordinate_list = self.__get_all_plist(self.user_tr_dict)
        # 获取每个轨迹点的聚类结果
        clusters = self.__gen_clusters(duration_list, coordinate_list, self.eps, self.duration)
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
        xy_list = []
        for idx, p in enumerate(tr):
            if idx == 0 or p.cluster_flag != tr[idx - 1].cluster_flag:
                if idx != 0:
                    tmp_p.coordinate = self.__get_core_coordinate(xy_list)
                    tmp_tr.append(tmp_p)
                tmp_p = p
                xy_list = [p.coordinate]
            else:
                tmp_p.duration += p.duration
                tmp_p.total_data += p.total_data
                tmp_p.time[1] = p.time[1]
                xy_list.append(p.coordinate)
        tmp_p.coordinate = self.__get_core_coordinate(xy_list)
        tmp_tr.append(tmp_p)
        return tmp_tr

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
