import random
from operator import itemgetter, attrgetter


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
    i_dict = {
        1: Stu('stu01', 22, 33),
        2: Stu('stu02', 33, 44),
        3: Stu('stu03', 44, 11),
        4: Stu('stu04', 55, 8),
        7: Stu('stu07', 44, 90),
        5: Stu('stu05', 66, 90),
        6: Stu('stu06', 44, 90)
    }
    # result = (sorted(i_dict.items(), key = lambda kv:(kv[1], kv[0])))
    # result = sorted(i_dict.items(), key=attrgetter('math'), reverse=True)
    result = (sorted(i_dict.items(), key=lambda kv: (kv[1], kv[0])))
    print(i_dict)
    for idx, val in enumerate(result):
        print(val[1].name, val[1].chinese, val[1].math)
        #print(type(val))


if __name__ == '__main__':
    # test_set()
    test_sort()
    # test_delete_element_from_list()
    # a = 111
    # b = 222
    # a, b = test_delete_element_from_list()
    # print(a, b)
    #test_merge_adjacent_points()