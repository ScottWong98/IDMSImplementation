from idms_impl.common.data_load import DataLoad
from idms_impl.common.stopover_area import StopoverAreaSet


class DataProcess:
    """数据处理
    将未处理的轨迹表转换成轨迹点为停留区域，并且具有多层语义的轨迹表
    """
    def __init__(self):
        # 轨迹表
        self.tr_dict = None
        self.__stopover_area_set = None

    def load_data(self, filename):
        data_load = DataLoad()
        self.tr_dict = data_load.load_data(filename="../../resource/test_1_user_with_flag.csv")

    def stopover_area_mining(self, eps, min_duration):
        # 停留区域挖掘
        self.__stopover_area_set = StopoverAreaSet(self.tr_dict)
        self.__stopover_area_set.stopover_area_mining(eps=eps, min_duration=min_duration)
        self.tr_dict = self.__stopover_area_set.tr_dict

    def semantic_tag_conversion(self, n, theta, knn_model, poi_list):
        # 语义转化
        self.__stopover_area_set.semantic_tag_conversion(n=n, theta=theta, knn_model=knn_model, poi_list=poi_list)
        self.tr_dict = self.__stopover_area_set.tr_dict
