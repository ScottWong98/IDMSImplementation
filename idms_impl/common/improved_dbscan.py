from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


class ImprovedDBSCAN:
    """扩展的DBSCAN算法"""

    UNCLASSIFIED = False
    NOISE = 0

    def __init__(self, duration_list, coordinate_list, eps, min_duration):
        # 轨迹点驻留时长列表
        self.duration_list = duration_list
        # 轨迹点坐标列表
        self.coordinate_list = coordinate_list
        # 聚类半径
        self.eps = eps
        # 最小驻留时长
        self.min_duration = min_duration

    def gen_cluster(self):
        """根据轨迹点生成聚类
        :return 各轨迹点的轨迹标签列表
        """
        cluster_id = 1
        # 轨迹点的总数
        p_num = self.coordinate_list.shape[1]
        # 初始化聚类列表，所有点均是未分类
        cluster_list = [self.UNCLASSIFIED] * p_num

        for pid in range(p_num):
            if cluster_list[pid] == self.UNCLASSIFIED:
                if self.__expand_cluster(cluster_list, pid, cluster_id):
                    cluster_id += 1
        return cluster_list

    def __expand_cluster(self, cluster_list, pid, cluster_id):
        """判断是否可以产生一个新的聚类，并对更新聚类里的点的聚类状态
        :param cluster_list: 所有点的聚类状态列表
        :param pid: 当前点的下标
        :param cluster_id: 当前的聚类标签
        :return 当前点是否是一个核心点
        """
        # 获取当前点邻域内的所有点的下标集合
        pid_list = self.__region_query_kd(pid)

        if self.__duration_sum(pid_list) < self.min_duration:
            # 无效点
            cluster_list[pid] = self.NOISE
            return False
        else:
            # 当前点是一个核心点
            cluster_list[pid] = cluster_id
            # 更新核心点邻域内的所有点
            while len(pid_list) > 0:
                cur_p = pid_list[0]
                query_res = self.__region_query_kd(cur_p)
                if self.__duration_sum(query_res) >= self.min_duration:
                    for i in range(len(query_res)):
                        res_p = query_res[i]
                        if cluster_list[res_p] == self.UNCLASSIFIED:
                            cluster_list[res_p] = cluster_id
                            pid_list.append(res_p)
                        elif cluster_list[res_p] == self.NOISE:
                            cluster_list[res_p] = cluster_id
                pid_list = pid_list[1:]
            return True

    def __region_query_kd(self, pid):
        """利用kdtree查询距离某点eps距离内的所有点的下标列表
        :param pid: 当前轨迹点在轨迹点列表中的下标
        :return 邻域内的所有点下标集合
        """
        t_coordinate_list = self.coordinate_list.transpose()
        kd_tree = KDTree(t_coordinate_list)
        return kd_tree.query_radius(t_coordinate_list[pid], r=self.eps)[0].tolist()

    def __duration_sum(self, pid_list):
        """计算距离当前点eps范围内的所有点的驻留总时长
        :param pid_list: 邻域内的所有点下标集合
        :return duration_sum: 驻留总时长
        """
        duration_sum = 0
        for i in pid_list:
            duration_sum += self.duration_list[i]
        return duration_sum
