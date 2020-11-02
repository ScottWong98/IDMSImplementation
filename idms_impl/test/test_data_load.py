from idms_impl.common.data_load import DataLoad, Trajectory


def test_load_data():
    print()
    data_load = DataLoad()
    tr_dict = data_load.load_data(filename="../../resource/test_12_user_with_flag.csv")
    trajectory = Trajectory(tr_dict)
    trajectory.output()