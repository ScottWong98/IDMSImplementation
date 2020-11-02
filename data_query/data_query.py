from data_processing.data_process import DataProcess
from data_store.index_tree import IndexTree
import time
import math


class IDMS:

    def __init__(self):
        self.origin_point = None
        self.side_length = None

    def set_grid(self, origin_point, side_length):
        self.origin_point = origin_point
        self.side_length = side_length

    def data_process(self, filename, eps, min_duration):
        data_process = DataProcess(filename, eps, min_duration)
        data_process.run()
        space_first_edges, semantic_first_edges = data_process.extract_edges(self.origin_point, self.side_length)
        return space_first_edges, semantic_first_edges

    @classmethod
    def data_store(cls, space_first_edges, semantic_first_edges):
        space_first_tree = IndexTree(raw_edges=space_first_edges)
        space_first_tree.build()
        semantic_first_tree = IndexTree(raw_edges=semantic_first_edges)
        semantic_first_tree.build()
        return space_first_edges, semantic_first_edges

    def data_query(self, filename, eps, min_duration, beta, space_first_tree, semantic_first_tree):
        query_data_process = DataProcess(filename, eps, min_duration)
        query_data_process.run()
        query_space_fist_edges, query_semantic_first_edges = query_data_process.extract_edges(
            self.origin_point,
            self.side_length
        )
        if beta >= 0.5:
            query_space_first_tree = IndexTree(raw_edges=query_space_fist_edges)
            query_space_first_tree.build()
        # query_space_first_tree.output()
            similar_tr = space_first_tree.query(query_space_first_tree.rt)
        else:
            query_semantic_first_tree = IndexTree(raw_edges=query_semantic_first_edges)
            query_semantic_first_tree.build()
            similar_tr = semantic_first_tree.query(query_semantic_first_tree.rt)

        print(similar_tr)
        return similar_tr


def simds(tr1, tr2, beta):
    coord1 = get_coordinate(tr1)
    coord2 = get_coordinate(tr2)
    h = space_similarity(coord1, coord2)
    print(h)

    # print(tr1)
    # print(tr2)

def get_coordinate(tr):
    res = []
    for (idx, point) in enumerate(tr):
        res.append((point.longitude, point.latitude))
    return res


def space_similarity(tr1, tr2):
    h1, d_max = get_h(tr1, tr2)
    h2, d_max = get_h(tr2, tr1)
    h = max(h1, h2)
    return 1 - h / d_max


def get_h(tr1, tr2):
    d_max, h = 0, 0
    for (idx1, p1) in enumerate(tr1):
        min_d = float("inf")
        for (idx2, p2) in enumerate(tr2):
            tmp_d = dist(p1, p2)
            d_max = max(d_max, tmp_d)
            min_d = min(min_d, tmp_d)
        h = max(h, min_d)
    return h, d_max


def dist(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def semantic_similarity(tr1, tr2):
    n, m = len(tr1), len(tr2)
    minn, maxx = min(n, m), max(n, m)
    sim = [minn / maxx]
    for i in range(maxx):
        sem1 = tr1[i] if i < n else []
        sem2 = tr2[i] if i < m else []
        sim.append(lcss(sem1, sem2) / minn)
    if sim[0] == 0 or sim[-1] == 0:
        return 0

    ans = 0.
    sum = maxx * (maxx + 1) / 2
    cur = maxx
    addon = 0.5
    sum += addon
    tt = 0.
    for (idx, val) in enumerate(sim):
        if idx == 0:
            ans += addon / sum * sim[0]
            tt += addon / sum
        else:
            ans += cur / sum * sim[idx]
            tt += cur / sum
            cur -= 1
    return ans


def lcss(sem1, sem2):
    if len(sem1) == 0 or len(sem2) == 0:
        return 0
    if sem1[0] == sem2[0]:
        return 1 + lcss(sem1[1:], sem2[1:])
    return max(lcss(sem1[1:], sem2), lcss(sem1, sem2[1:]))


if __name__ == '__main__':

    start = time.process_time()
    data_process = DataProcess("../resource/test_12_user_with_flag.csv", 0.01, 5000)
    data_process.run()
    space_first_edges, semantic_first_edges = data_process.extract_edges(origin_point=(190, 190), side_length=5)
    space_first_tree = IndexTree(raw_edges=space_first_edges)
    space_first_tree.build()
    semantic_first_tree = IndexTree(raw_edges=semantic_first_edges)
    semantic_first_tree.build()

    query_data_process = DataProcess("../resource/test_1_user_1_day_with_flag.csv", 0.01, 5000)
    query_data_process.run()
    query_space_fist_edges, query_semantic_first_edges = data_process.extract_edges(origin_point=(190, 190),
                                                                                    side_length=5)
    query_space_first_tree = IndexTree(raw_edges=query_space_fist_edges)
    query_space_first_tree.build()
    # query_space_first_tree.output()

    similar_tr = space_first_tree.query(query_space_first_tree.rt)
    print('*' * 60)
    print(list(query_data_process.user_dict.values())[0].output())
    query_tr = list(list(query_data_process.user_dict.values())[0].tr_dict.values())[0].plist
    print(query_tr)
    print(type(query_tr))
    # for (idx, point) in enumerate(query_tr):
    #     print(point)
    user_dict = data_process.user_dict
    for tr in similar_tr:
        simds(query_tr, user_dict[tr[0]].tr_dict[tr[1]].plist, 0.4)
        # print(simds(query_data_process.user_dict))
        # print(data_process.user_dict[tr[0]].tr_dict[tr[1]].output())
        # print(len(data_process.user_dict[tr[0]].tr_dict[tr[1]].plist))
    # print(similar_tr)
    # print(data_process.user_dict["178464"].tr_dict["20200116"].output())
    # print(len(similar_tr))
    # print(data_process.user_dict["178464"].output())
    # print(query_data_process.user_dict)
    end = time.process_time()
    print("Finish all in %s " % str(end - start))
