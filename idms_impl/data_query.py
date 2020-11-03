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
        result_similar_tr = []
        for tr_id in similar_tr:
            user_id, date = tr_id[0], tr_id[1]
            tr = self.tr_dict[user_id][date]
            value = simds.simds(tr, query_tr, beta)
            if user_id not in result_tr_dict:
                result_tr_dict[user_id] = {}
            result_tr_dict[user_id][date] = [tr, value]
            result_similar_tr.append(SimilarTr(user_id, date, tr, value))
        result_similar_tr = sorted(result_similar_tr)
        print('\n' * 3)
        print('=' * 100)
        print('The query result is :')
        for res in result_similar_tr:
            if res.value < theta:
                break
            res.output()

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


class SimilarTr:

    def __init__(self, user_id, date, tr, value):
        self.user_id = user_id
        self.date = date
        self.tr = tr
        self.value = value

    def __lt__(self, other):
        return self.value >= other.value

    def output(self):
        print('=' * 50)
        print('UserID:', self.user_id)
        print('=' * 50)
        print('------------- Date: %s -----------' % self.date)
        print('------------- Value: %s -----------' % self.value)
        for p in self.tr:
            print(p)

