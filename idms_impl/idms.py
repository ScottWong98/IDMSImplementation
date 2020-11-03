from idms_impl.data_process import DataProcess
from idms_impl.data_storage import DataStorage
from idms_impl.data_query import DataQuery
from idms_impl.common.poi import POILoad
from idms_impl.common.data_load import tr_dict_output
from idms_impl.common.index_tree import IndexTree
import time


class IDMS:

    def __init__(self):
        self.tr_dict = None
        self.query_tr_dict = None
        self.poi_list = None
        self.knn_model = None
        self.index_tree = None
        self.eps = None
        self.min_duration = None
        self.n = None
        self.theta = None
        self.origin_point = None
        self.side_length = None

    def load_poi(self, filename, k):
        poi_load = POILoad(filename)
        poi_load.train_data(k=k)
        self.knn_model, self.poi_list = poi_load.knn_model, poi_load.poi_list

    def process_data(self, filename):
        data_process = DataProcess()
        data_process.load_data(filename)
        data_process.stopover_area_mining(eps=self.eps, min_duration=self.min_duration)
        data_process.semantic_tag_conversion(n=self.n, theta=self.theta, knn_model=self.knn_model, poi_list=self.poi_list)
        self.tr_dict = data_process.tr_dict

    def storage_data(self):
        data_storage = DataStorage(self.tr_dict)
        data_storage.build_index_tree(self.origin_point, self.side_length)
        self.index_tree = data_storage.tree

    def query_data(self, filename, beta, theta):
        # 预处理出待查询轨迹
        query_data_process = DataProcess()
        query_data_process.load_data(filename)
        query_data_process.stopover_area_mining(eps=self.eps, min_duration=self.min_duration)
        query_data_process.semantic_tag_conversion(n=self.n, theta=self.theta, knn_model=self.knn_model, poi_list=self.poi_list)
        query_tr_dict = query_data_process.tr_dict

        # 构建待查询轨迹树，包含语义优先和空间优先
        query_tree = IndexTree(query_tr_dict)
        query_tree.build(origin_point=self.origin_point, side_length=self.side_length)

        # 初始化DataQuery
        data_query = DataQuery(self.tr_dict, self.index_tree)

        # 得到查询轨迹列表形式
        query_tr = list(list(query_tr_dict.values())[0].values())[0]

        print("=" * 100)
        print("The query trajectory is :")
        tr_dict_output(query_tr_dict)

        # 进行查询
        data_query.query(query_tree, query_tr, beta, theta)


if __name__ == '__main__':
    start_time = time.time()

    idms = IDMS()
    # 设置相关参数
    idms.eps = 0.01
    idms.min_duration = 5000
    idms.n = 22
    idms.theta = 0.9
    idms.origin_point = (190, 190)
    idms.side_length = 5

    # 加载POI文件
    idms.load_poi("../resource/nj_poi.csv", k=6)

    idms.process_data(filename="../resource/test_12_user_with_flag.csv")
    idms.storage_data()
    idms.query_data(filename="../resource/test_user_1_day_22_with_flag.csv",
                    beta=0.9,
                    theta=0.5)

    end_time = time.time()
    print("Finish it in %s s" % (end_time - start_time))