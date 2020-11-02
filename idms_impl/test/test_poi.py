from idms_impl.common.poi import POILoad


def test_poi_load():
    print()
    poi_load = POILoad(filename="../../resource/nj_poi.csv")
    poi_load.train_data(k=5)
    # for poi in poi_load.poi_list:
    #     print(poi)
    # print(poi_load.poi_list)
    print(poi_load.knn_model)
