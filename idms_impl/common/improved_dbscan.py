from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt


class ImprovedDBSCAN:
    """扩展的DBSCAN算法"""

    UNCLASSIFIED = False
    NOISE = 0

    def __init__(self, duration_list, coordinate_list, eps, min_duration):
        self.duration_list = duration_list
        self.coordinate_list = coordinate_list
        self.eps = eps
        self.min_duration = min_duration

    def gen_cluster(self):
        """根据轨迹点生成聚类"""
        cluster_id = 1
        p_num = self.coordinate_list.shape[1]
        # 初始化聚类列表
        cluster_list = [self.UNCLASSIFIED] * p_num
        for pid in range(p_num):
            if cluster_list[pid] == self.UNCLASSIFIED:
                if self.__expand_cluster(cluster_list, pid, cluster_id):
                    cluster_id += 1
        return cluster_list

    def __expand_cluster(self, cluster_list, pid, cluster_id):
        pid_list = self.__region_query_kd(pid)
        if self.__duration_sum(pid_list) < self.min_duration:
            cluster_list[pid] = self.NOISE
            return False
        else:
            cluster_list[pid] = cluster_id
            while len(pid_list) > 0:
                cur_p = pid_list[0]
                query_res = self.__region_query_kd(cur_p)
                if self.__duration_sum(query_res) >= self.min_duration:
                    for i in range(len(query_res)):
                        res_p = query_res[i]
                        if cluster_list[res_p] == self.UNCLASSIFIED:
                            cluster_list[res_p] = cluster_id
                        elif cluster_list[res_p] == self.NOISE:
                            cluster_list[res_p] = cluster_id
                pid_list = pid_list[1:]
            return True

    def __region_query_kd(self, pid):
        """利用kdtree查询距离某点eps距离内的所有点的下标列表"""
        t_coordinate_list = self.coordinate_list.transpose()
        kd_tree = KDTree(t_coordinate_list)
        return kd_tree.query_radius(t_coordinate_list[pid], r=self.eps)[0].tolist()

    def __duration_sum(self, pid_list):
        duration_sum = 0
        for i in pid_list:
            duration_sum += self.duration_list[i]
        return duration_sum
