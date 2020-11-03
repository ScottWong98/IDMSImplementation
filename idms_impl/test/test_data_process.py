from idms_impl.common.data_load import DataLoad, tr_dict_output
from idms_impl.common.poi import POILoad
from idms_impl.data_process import DataProcess


"""
def test_stopover_mining():
    print()
    # 加载数据
    data_load = DataLoad()
    tr_dict = data_load.load_data(filename="../../resource/test_1_user_with_flag.csv")
    # 停留区域挖掘
    stopover_area_set = StopoverAreaSet(tr_dict)
    stopover_area_set.stopover_area_mining(eps=0.01, min_duration=5000)
    # 语义转化
    poi_load = POILoad(filename="../../resource/nj_poi.csv")
    poi_load.train_data(k=5)
    knn_model, poi_list = poi_load.knn_model, poi_load.poi_list
    stopover_area_set.semantic_tag_conversion(n=22, theta=0.99, knn_model=knn_model, poi_list=poi_list)
    tr_dict = stopover_area_set.tr_dict

    tr_dict_output(tr_dict)
"""


def test_data_process():
    print()
    data_process = DataProcess()
    data_process.load_data(filename="../../resource/test_1_user_with_flag.csv")
    data_process.stopover_area_mining(eps=0.01, min_duration=5000)
    tr_dict_output(data_process.tr_dict)
    # poi_load = POILoad(filename="../../resource/nj_poi.csv")
    # poi_load.train_data(k=5)
    # knn_model, poi_list = poi_load.knn_model, poi_load.poi_list
    # data_process.semantic_tag_conversion(n=22, theta=0.99, knn_model=knn_model, poi_list=poi_list)
    # tr_dict = data_process.tr_dict
    #
    # tr_dict_output(tr_dict)