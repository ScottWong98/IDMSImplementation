import math


class SIMDS:

    def simds(self, tr1, tr2, beta):
        xy_list1 = self.__extract_coordinate(tr1)
        xy_list2 = self.__extract_coordinate(tr2)
        # print(xy_list1)
        # print(xy_list2)
        d = self.__space_similarity(xy_list1, xy_list2)
        # print("d: ", d)

        sem_dict1 = self.__extract_semantic(tr1)
        sem_dict2 = self.__extract_semantic(tr2)
        # print(sem_dict1)
        # print(sem_dict2)
        s = self.__semantic_similarity(sem_dict1, sem_dict2)

        # print("s: ", s)
        return beta * d + (1 - beta) * s

    @classmethod
    def __space_similarity(cls, xy_list1, xy_list2):
        h1, d_max1 = cls.__get_h(xy_list1, xy_list2)
        # print("h1: %s, d_max1: %s" % (h1, d_max1))
        h2, d_max2 = cls.__get_h(xy_list2, xy_list1)
        # print("h2: %s, d_max2: %s" % (h2, d_max2))
        d_max = max(d_max1, d_max2)
        # print("d_max: ", d_max)
        h = max(h1, h2)
        # print("h: ", h)
        return 1. - h / d_max

    @classmethod
    def __semantic_similarity(cls, sem_dict1, sem_dict2):
        n, m = len(sem_dict1), len(sem_dict2)
        minn, maxx = min(n, m), max(n, m)
        sim = [minn / maxx]
        for i in reversed(range(maxx)):
            sem1 = sem_dict1[i] if i in sem_dict1 else []
            sem2 = sem_dict2[i] if i in sem_dict2 else []
            # print(cls.__lcss(sem1, sem2))
            sim.append(cls.__lcss(sem1, sem2) / minn)
        # print(sim)
        if sim[0] == 0 or sim[-1] == 0:
            return 0

        ans = 0
        sum = maxx * (maxx + 1) / 2
        cur = maxx
        addon = 0.5
        sum += addon
        tt = 0
        for idx, val in enumerate(sim):
            if idx == 0:
                ans += addon / sum * sim[0]
                tt += addon/ sum
            else:
                ans += cur / sum * sim[idx]
                tt += cur / sum
                cur -= 1
        return ans

    @classmethod
    def __lcss(cls, sem1, sem2):
        if len(sem1) == 0 or len(sem2) == 0:
            return 0
        if sem1[0] == sem2[0]:
            return 1 + cls.__lcss(sem1[1:], sem2[1:])
        return max(cls.__lcss(sem1[1:], sem2), cls.__lcss(sem1, sem2[1:]))

    @classmethod
    def __get_h(cls, xy_list1, xy_list2):
        d_max, h = 0, 0
        for idx1, p1 in enumerate(xy_list1):
            min_d = float('inf')
            for idx2, p2 in enumerate(xy_list2):
                tmp_d = cls.__dist(p1, p2)
                d_max = max(d_max, tmp_d)
                min_d = min(min_d, tmp_d)
            h = max(h, min_d)
        return h, d_max

    @classmethod
    def __dist(cls, xy_1, xy_2):
        return math.sqrt((xy_1[0] - xy_2[0]) ** 2 + (xy_1[1] - xy_2[1]) ** 2)

    @classmethod
    def __extract_coordinate(cls, tr):
        res = []
        for idx, p in enumerate(tr):
            res.append(p.coordinate)
        return res

    @classmethod
    def __extract_semantic(cls, tr):
        res = {}
        for p in tr:
            for idx, sem in enumerate(p.sr):
                if idx not in res:
                    res[idx] = []
                res[idx].append(sem)
        return res




