import random


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


if __name__ == '__main__':
    test_set()
    # test_delete_element_from_list()
    # a = 111
    # b = 222
    # a, b = test_delete_element_from_list()
    # print(a, b)
    #test_merge_adjacent_points()