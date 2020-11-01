import random
from operator import itemgetter, attrgetter
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


def test_delete_element_from_list():
    i_list = [1, 1, 1, 1, 2, 2, 3, 4, 4]
    j_list = []
    tmp = 0
    for idx, val in enumerate(i_list):

        print('*' * 10)
        print(idx, val)
        print(i_list)
        if idx == 0 or val != i_list[idx - 1]:
            if idx != 0:
                j_list.append(tmp)
            tmp = val
        else:
            tmp += val
        print(i_list)
        print('*' * 10)
    j_list.append(tmp)
    print(j_list)


def test_merge_adjacent_points():
    # random.seed(0)
    # a = ['1' if random.random() > 0.4 else ' ' for i in range(10)]
    # b = [len(x) for x in ''.join(a).split()]
    # c = ''.join(a).split()
    #
    # print(a)
    #
    # print(b)
    # print(c)
    i_list = [1, 1, 1, 2, 3, 3, 3, 5, 4, 4, 4, 1, 1, 1]
    flag = False
    result = {}
    for idx, val in enumerate(i_list):
        if idx == 0 or val != i_list[idx - 1]:
            result[val] = 1
        else:
            result[val] += 1
    print(result)


def test_set():
    i_set = set()
    i_set.add(1)
    i_set.add(2)
    i_set.add(1)
    i_set.add(3)
    for i in i_set:
        print(i)
    print(i_set)


class Stu:

    def __init__(self, name, chinese, math):
        self.name = name
        self.chinese = chinese
        self.math = math

    def __lt__(self, other):
        return self.math >= other.math


def test_sort():
    # i_dict = {
    #     1: Stu('stu01', 22, 33),
    #     2: Stu('stu02', 33, 44),
    #     3: Stu('stu03', 44, 11),
    #     4: Stu('stu04', 55, 8),
    #     7: Stu('stu07', 44, 90),
    #     5: Stu('stu05', 66, 90),
    #     6: Stu('stu06', 44, 90)
    # }
    # # result = (sorted(i_dict.items(), key = lambda kv:(kv[1], kv[0])))
    # # result = sorted(i_dict.items(), key=attrgetter('math'), reverse=True)
    # result = (sorted(i_dict.items(), key=lambda kv: (kv[1], kv[0])))
    # print(i_dict)
    # for idx, val in enumerate(result):
    #     print(val[1].name, val[1].chinese, val[1].math)

    i_set = set()
    # print(i_set)
    i_set.union({1, 2})
    i_set |= {1, 2}
    print(i_set)
    i_dict = {
        '住宿': 123,
        '交通': 1231,
        '体育': 12311,
        '公共设施': 22
    }

    a = 1
    b = 2
    if a == b:
        rs = 234
    else:
        rs = 333
    print(rs)
    #
    # i_dict = {
    #     (1, 2): 123,
    #     (1, 1): 1231,
    #     (2, 3): 12311,
    #     (1, 3): 22
    # }

    # i_dict = {
    #     'rT': 123,
    #     'wer': 123,
    #     'a': 123
    # }
    # if isinstance(list(i_dict.keys())[0], tuple):
    #     print(12312312)
    # print(type(list(i_dict.keys())[0]))

    result = sorted(i_dict.items(), key=lambda d:d[0].encode('gbk'))
    print(result)
    # for item in result:
    #     pass
    # result = {}
    # for key in sorted(i_dict.keys()):
    #     result[key] = i_dict[key]
    # i_dict = result
    # print(i_dict)



def test_KNN():
    X = [[0], [1], [1], [2], [3]]
    y = [0, 0, 0, 1, 1]
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(X, y)
    print(neigh.predict([[1.1]]))
    print(neigh.kneighbors([[1.1]]))


def test_counter():
    List = [1, 2, 2, 4, 5]
    seed = 6
    match_num = [i for i in List if i == seed]
    print(match_num)
    if len(match_num) == 0:
        print("1231231")

    i_dict = {
        "1": 1,
        "2": 2
    }
    print(list(i_dict.items())[0][0])


def test_list():
    i_dict = {
        "RT": '12312312',
        (1, 2): "12312312",
        "adfa": "adsfasdfa",
        "asdfasf": "asdf"
    }
    for (key, value) in i_dict.items():
        print(key, value)


if __name__ == '__main__':
    # test_list()
    #test_counter()
    # test_KNN()
    # test_set()
    test_sort()
    # test_delete_element_from_list()
    # a = 111
    # b = 222
    # a, b = test_delete_element_from_list()
    # print(a, b)
    #test_merge_adjacent_points()