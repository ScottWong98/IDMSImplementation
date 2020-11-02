from idms_impl.common.stopover_area import StopoverAreaSet
from idms_impl.common.data_load import DataLoad, tr_dict_output


def test_stopover_mining():
    print()
    data_load = DataLoad()
    tr_dict = data_load.load_data(filename="../../resource/test_1_user_with_flag.csv")
    stopover_area_set = StopoverAreaSet(tr_dict)
    stopover_area_set.stopover_area_mining(eps=0.01, min_duration=5000)
    tr_dict = stopover_area_set.tr_dict
    tr_dict_output(tr_dict)