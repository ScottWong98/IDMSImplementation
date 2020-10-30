import time
from data_processing.data_load import DataLoad
from data_processing.poi import POILibrary

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

    def stopover_area_excavation(self):
        """停留点区域挖掘
        :return: 带有聚类标签的用户轨迹数据
        """
        # 1. 加载数据，返回的是一个以user_id为主键的User字典
        data_load = DataLoad(self.filename)
        self.user_dict = data_load.load_data()

        # 2. 遍历每一个用户，进行停留点区域挖掘
        for (user_id, user) in self.user_dict.items():
            user.stopover_area_excavation(self.eps, self.min_duration)

    def semantic_tag_conversion(self, n, theta, knn_model, poi_list):
        # 1. 遍历每一个用户，进行语义转化
        for (user_id, user) in self.user_dict.items():
            user.semantic_tag_conversion(n, theta, knn_model, poi_list)
            user.output()


if __name__ == '__main__':
    start = time.process_time()
    # 1. 加载用户文件
    data_process = DataProcess("../resource/test_input_1.csv", 0.01, 5000)

    # 2. 停留区域挖掘
    data_process.stopover_area_excavation()

    # 3. 针对百度POI建立KNN模型
    poi_library = POILibrary()
    neigh, poi_list_ = poi_library.generate_knn_model()

    # 4. 停留区域多层次语义标签转换
    data_process.semantic_tag_conversion(n=22, theta=0.99, knn_model=neigh, poi_list=poi_list_)

    end = time.process_time()
    print("Finish all in %s " % str(end - start))

