import time
import math
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
        # 栅格化所选取的原点
        self.origin_point = ()
        # 栅格化所选取的边长
        self.side_length = 0

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

    def run(self):
        self.stopover_area_excavation()
        poi_library = POILibrary()
        neigh, poi_list_ = poi_library.generate_knn_model()
        self.semantic_tag_conversion(n=22, theta=0.99, knn_model=neigh, poi_list=poi_list_)

    def extract_edges(self, origin_point, side_length):
        self.origin_point = origin_point
        self.side_length = side_length
        space_first_edges = []
        semantic_first_edges = []
        for user_id, user in self.user_dict.items():
            for date, tr in user.tr_dict.items():
                tr_id = (user.user_id, date)
                for (idx, point) in enumerate(tr.plist):
                    xx, yy = self.get_new_coordinate(point.longitude, point.latitude)
                    code_id = (xx, yy)
                    _space_first_list = ['RT', code_id]
                    _sem_first_list = ['RT']
                    for element in point.sr:
                        _space_first_list.append(element)
                        _sem_first_list.append(element)
                    _sem_first_list.append(code_id)

                    space_first_edges = self.update_edges(_space_first_list, space_first_edges, tr_id)
                    semantic_first_edges = self.update_edges(_sem_first_list, semantic_first_edges, tr_id)
        return space_first_edges, semantic_first_edges

    @classmethod
    def update_edges(cls, _list, edges, tr_id):
        for (depth, sp) in enumerate(_list):
            if depth == len(_list) - 1:
                break
            if depth >= len(edges):
                edges.append({})
            if sp not in edges[depth]:
                edges[depth][sp] = {}
            ep = _list[depth + 1]
            if ep not in edges[depth][sp]:
                edges[depth][sp][ep] = set()
            edges[depth][sp][ep].add(tr_id)
        return edges

    def get_new_coordinate(self, x, y):
        xx = math.floor((x - self.origin_point[0]) / self.side_length)
        yy = math.floor((y - self.origin_point[1]) / self.side_length)
        return xx, yy


if __name__ == '__main__':
    start = time.process_time()
    # # 1. 加载用户文件
    # data_process = DataProcess("../resource/test_13_user_with_flag.csv", 0.01, 5000)
    #
    # # 2. 停留区域挖掘
    # data_process.stopover_area_excavation()
    #
    # # 3. 针对百度POI建立KNN模型
    # poi_library = POILibrary()
    # neigh, poi_list_ = poi_library.generate_knn_model()
    #
    # # 4. 停留区域多层次语义标签转换
    # data_process.semantic_tag_conversion(n=22, theta=0.99, knn_model=neigh, poi_list=poi_list_)
    data_process = DataProcess("../resource/test_13_user_with_flag.csv", 0.01, 5000)
    data_process.run()
    space_first_edges, semantic_first_edges = data_process.extract_edges(origin_point=(190, 190), side_length=5)
    # print(space_first_edges)
    # print(semantic_first_edges)

    end = time.process_time()
    print("Finish all in %s " % str(end - start))

