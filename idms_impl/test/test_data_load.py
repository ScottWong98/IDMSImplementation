from idms_impl.common.data_load import DataLoad, tr_dict_output


def test_load_data():
    print()
    data_load = DataLoad()
    tr_dict = data_load.load_data(filename="../../resource/test_12_user_with_flag.csv")
    tr_dict_output(tr_dict)