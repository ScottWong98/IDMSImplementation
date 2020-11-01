from data_processing.data_process import DataProcess
from data_processing.poi import POILibrary
from my_data_storage.index_tree import IndexTree
import time
import math


class DataStorage:
    """对数据进行存储"""
    def __init__(self, user_dict):
        # 轨迹表
        self.user_dict = user_dict
        # 索引树
        self.index_tree = None
        # 栅格化所选取的原点
        self.origin_point = ()
        # 栅格化所选取的边长
        self.side_length = 0
        # 空间语义所占比例
        self.ratio = 0

    def set_grid(self, origin_point, side_length):
        # 栅格化所选取的原点
        self.origin_point = origin_point
        # 栅格化所选取的边长
        self.side_length = side_length

    def set_ratio(self, ratio):
        self.ratio = ratio

    def storage(self):
        pass

    def build(self):
        """构建IDMS索引表"""
        space_first_edges, semantic_first_edges = self.generate_edges()
        # print(space_first_edges)
        # print(semantic_first_edges)
        # print(se)
        space_first_tree = IndexTree(space_first_edges)
        space_first_tree.build()
        semantic_first_tree = IndexTree(semantic_first_edges)
        semantic_first_tree.build()
        # print("Finish tree build...........")
        semantic_first_tree.output()
        print((len(semantic_first_tree.rt.tr_id_set)))
        # semantic_first_
        # print(len(space_first_tree.rt.tr_id_set))

    def generate_edges(self):
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

    # 1. 加载用户文件
    data_process = DataProcess("../resource/test_input_1.csv", eps=0.01, min_duration=5000)

    # 2. 停留区域挖掘
    data_process.stopover_area_excavation()

    # 3. 针对百度POI建立KNN模型
    poi_library = POILibrary()
    neigh, poi_list_ = poi_library.generate_knn_model()

    # 4. 停留区域多层次语义标签转换
    data_process.semantic_tag_conversion(n=22, theta=0.99, knn_model=neigh, poi_list=poi_list_)

    # 5. 对数据进行存储
    data_storage = DataStorage(data_process.user_dict)
    data_storage.set_grid(origin_point=(190, 190), side_length=5)
    data_storage.build()

    end = time.process_time()
    print("Finish all in %s " % str(end - start))
