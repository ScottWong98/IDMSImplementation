from idms_impl.data_process import DataProcess
from idms_impl.data_storage import DataStorage
from idms_impl.data_query import DataQuery
from idms_impl.common.poi import POILoad


class IDMS:

    def __init__(self):
        self.tr_dict = None
        self.query_tr_dict = None
        self.poi_list = None
        self.knn_model = None

    def load_poi(self, filename, k):
        poi_load = POILoad(filename)
        poi_load.train_data(k=k)
        self.knn_model, self.poi_list = poi_load.knn_model, poi_load.poi_list

    def process_data(self, filename, eps, min_duration, n, theta):
        data_process = DataProcess()
        data_process.load_data(filename)
        data_process.stopover_area_mining(eps=eps, min_duration=min_duration)
        data_process.semantic_tag_conversion(n=n, theta=theta, knn_model=self.knn_model, poi_list=self.poi_list)
        self.tr_dict = data_process.tr_dict

    def storage_data(self, origin_point, side_length):
        data_storage = DataStorage(self.tr_dict)
        data_storage.build_index_tree(origin_point, side_length)
        pass
