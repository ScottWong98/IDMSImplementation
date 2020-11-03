from idms_impl.common.data_load import DataLoad
from idms_impl.common.stopover_area import StopoverAreaMining, SemanticTagConversion


class DataProcess:
    """数据处理
    将文件中的数据转化成带有多层语义的轨迹表
    """
    def __init__(self):
        # 轨迹表
        self.tr_dict = None

    def load_data(self, filename):
        """加载数据
        将数据加载到 `self.tr_dict` 中
        :param filename: 文件名
        """
        data_load = DataLoad()
        self.tr_dict = data_load.load_data(filename=filename)

    def stopover_area_mining(self, eps, min_duration):
        """停留区域挖掘
        更新 `self.tr_dict`，将每个轨迹点的 `cluster_flag` 更新为相应的聚类标签
        :param eps: 半径长度
        :param min_duration: 最小驻留时长
        """
        # 停留区域挖掘
        stopover_area_mining = StopoverAreaMining(self.tr_dict, eps, min_duration)
        stopover_area_mining.run()

    def semantic_tag_conversion(self, n, theta, knn_model, poi_list):
        """轨迹点的语义标签转化
        更新 `self.tr_dict`，将每个轨迹点的 `sr` 更新为相应的多层POI信息
        :param n: 用户行为的观察天数
        :param theta: 提取SR时，duration累计占比的阈值
        :param knn_model: 对POI信息训练好的KNN模型
        :param poi_list: POI列表
        """
        # 语义转化
        semantic_tag_conversion = SemanticTagConversion(self.tr_dict, n, theta, knn_model, poi_list)
        semantic_tag_conversion.run()
