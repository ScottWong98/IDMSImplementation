from idms_impl.data_query import DataQuery
from idms_impl.common.index_tree import tree_output
from idms_impl.data_storage import DataStorage
from idms_impl.common.data_load import DataLoad, tr_dict_output
from idms_impl.common.poi import POILoad
from idms_impl.data_process import DataProcess


def test_data_query():
    print()
    data_process = DataProcess()
    data_process.load_data(filename="../../resource/test_12_user_with_flag.csv")
    data_process.stopover_area_mining(eps=0.01, min_duration=5000)
    poi_load = POILoad(filename="../../resource/nj_poi.csv")
    poi_load.train_data(k=6)
    knn_model, poi_list = poi_load.knn_model, poi_load.poi_list
    data_process.semantic_tag_conversion(n=22, theta=0.9, knn_model=knn_model, poi_list=poi_list)
    tr_dict = data_process.tr_dict

    # tr_dict_output(tr_dict)

    data_storage = DataStorage(tr_dict)
    spf_tree, sef_tree = data_storage.build_index_tree(origin_point=(190, 190), side_length=5)
    # tree_output(spf_tree)
    # tree_output(sef_tree)

    data_process.load_data(filename="../../resource/test_user_1_day_22_with_flag.csv")
    data_process.stopover_area_mining(eps=0.01, min_duration=5000)
    data_process.semantic_tag_conversion(n=22, theta=0.9, knn_model=knn_model, poi_list=poi_list)
    query_tr_dict = data_process.tr_dict
    query_tr = list(list(query_tr_dict.values())[0].values())[0]
    print("=" * 100)
    print("The query trajectory is :")
    tr_dict_output(query_tr_dict)

    query_data_storage = DataStorage(query_tr_dict)
    query_data_storage.build_index_tree(origin_point=(190, 190), side_length=5)

    query_tree = query_data_storage.tree

    data_query = DataQuery(tr_dict, data_storage.tree)
    data_query.query(query_tree, query_tr, 0.9, 0.5)
