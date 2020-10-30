import time
import numpy as np

from data_processing.data_load import DataLoad
from data_processing.improved_dbscan import ImprovedDBSCAN


class DataProcess:
    """数据预处理模块
    将收集到的用户数据处理成具有语义的用户轨迹数据
    """
    def __init__(self, filename, eps, min_duration):
        """初始化
        :param filename: 读入的文件
        :param eps: 聚类中设置的半径
        :param min_duration: 聚类中设置的最小停留时间
        """
        self.filename = filename
        self.eps = eps
        self.min_duration = min_duration
        self.user_dict = {}

    # @classmethod
    # def get_user_plist(cls, user_dict):
    #     plist, loc_plist = [], []
    #     for (user_id, user) in user_dict.items():
    #         for (date, tr) in user.tr_dict.items():
    #             for idx, point in enumerate(tr.plist):
    #                 _list = [point.time_in, point.time_out, point.longitude, point.latitude,
    #                          point.duration, point.day_number, point.total_data]
    #                 plist.append(_list)
    #                 loc_plist.append([point.longitude, point.latitude])
    #     return plist, loc_plist

    def stopover_area_excavation(self):
        """停留点区域挖掘
        :return: 带有聚类标签的用户轨迹数据
        """
        # 1. 加载数据，返回的是一个以user_id为主键的User字典
        data_load = DataLoad()
        self.user_dict = data_load.load_data(self.filename)

        # 2. 遍历每一个用户，进行停留点区域挖掘
        for (user_id, user) in self.user_dict.items():
            user.stopover_area_excavation(self.eps, self.min_duration)

    # TODO
    def semantic_tag_conversion(self, n, theta):
        # 1. 遍历每一个用户，进行语义转化
        for (user_id, user) in self.user_dict.items():
            user.semantic_tag_conversion(n, theta)


if __name__ == '__main__':
    start = time.process_time()
    data_process = DataProcess("../resource/test_input_2.csv", 0.01, 5000)
    data_process.stopover_area_excavation()
    data_process.semantic_tag_conversion(n=22, theta=0.99)
    end = time.process_time()
    print("Finish all in %s " % str(end - start))

