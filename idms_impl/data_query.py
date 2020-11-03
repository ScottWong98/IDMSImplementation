from idms_impl.common.simds import SIMDS
from idms_impl.common.data_load import tr_dict_output


class DataQuery:

    def __init__(self, tr_dict, index_tree):
        self.tr_dict = tr_dict
        self.index_tree = index_tree

    def query(self, query_tree, query_tr, beta, theta):
        # 获取相似轨迹集合
        similar_tr = self.index_tree.query(query_tree, beta)

        simds = SIMDS()
        result_tr_dict = {}
        for tr_id in similar_tr:
            user_id, date = tr_id[0], tr_id[1]
            tr = self.tr_dict[user_id][date]
            value = simds.simds(tr, query_tr, beta)
            if user_id not in result_tr_dict:
                result_tr_dict[user_id] = {}
            result_tr_dict[user_id][date] = [tr, value]
        self.__output(result_tr_dict)

    @classmethod
    def __output(cls, result_tr_dict):
        for user_id, user_tr_dict in result_tr_dict.items():
            print('=' * 40)
            print('User ID: ', user_id)
            for date, tr_value in user_tr_dict.items():
                print('-' * 30)
                print('Date: ', date)
                for p in tr_value[0]:
                    print(p)
                print('Similar value: ', tr_value[1])




