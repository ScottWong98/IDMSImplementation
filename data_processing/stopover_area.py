from data_processing.poi import POILibrary


class StopoverArea:

    def __init__(self):
        self.sn = 0
        self.duration = 0
        self.d = 0.0
        # 停留区中心区域坐标
        self.coordinate = []
        # 停留区对应的POI类别
        self.category = []

    def __lt__(self, other):
        return self.duration >= other.duration


class StopoverAreaSet:

    def __init__(self, tr_dict, n, theta):
        # 用户所有轨迹集合，一个字典，可根据时间获取到当天的轨迹
        self.tr_dict = tr_dict
        # 停留区域字典，key为停留区域聚类标签，value为StopoverArea
        self.area_dict = {}
        self.n = n
        self.theta = theta
        self.sum_d = 0.0
        self.sum_duration = 0

    def get_semantic_dict(self):
        # 1. 获取主要驻留区域
        sr_dict = self.get_main_stopover_area()

        # 2. 利用百度POI获取到各点关联的POI信息
        poi_library = POILibrary(self.tr_dict, self.area_dict, sr_dict)
        semantic_dict = poi_library.get_sementic_dict()

        return semantic_dict

    def get_main_stopover_area(self):
        sr_dict = {}
        sr_list = []

        # 1. 处理各停留区的数据信息
        self.handle_all_area()
        # self.output()

        # 2. 对各停留区依据duration进行降序排列
        self.sort_in_des_order()
        # self.output()

        # 3. 提取dura累计占比在 theta 前的sr，存入sr_list
        tmp_sum_duration = 0.0
        for (cluster_id, area) in self.area_dict.items():
            tmp_sum_duration += area.duration
            # print(area.duration, tmp_sum_duration)
            # print(tmp_sum_duration / self.sum_duration)
            if tmp_sum_duration / self.sum_duration <= self.theta:
                sr_list.append(cluster_id)
        n_sr = len(sr_list)
        if n_sr == 1:
            sr_dict[sr_list[0]] = "HOME"
        elif n_sr == 2:
            idx1, idx2 = sr_list[0], sr_list[1]
            home_probe1 = self.home_probe(self.area_dict[idx1].sn, self.area_dict[idx1].d)
            home_probe2 = self.home_probe(self.area_dict[idx2].sn, self.area_dict[idx2].d)
            if home_probe1 > home_probe2:
                sr_dict[idx1], sr_dict[idx2] = "HOME", "WORK"
            else:
                sr_dict[idx1], sr_dict[idx2] = "WORK", "HOME"
        elif n_sr == 3:
            idx1, idx2, idx3 = sr_list[0], sr_list[1], sr_list[2]
            d1, d2, d3 = self.area_dict[idx1].d, self.area_dict[idx2].d, self.area_dict[idx3].d
            if d1 > (d2 + d3) / 2:
                sr_dict[idx1], sr_dict[idx2], sr_dict[idx3] = "WORK", "HOME", "HOME"
            else:
                sr_dict[idx1], sr_dict[idx2], sr_dict[idx3] = "HOME", "WORK", "WORK"

        return sr_dict

    def home_probe(self, sn, d):
        sn_probe = sn / self.n
        d_probe = d / self.sum_d
        return sn_probe + (1 - d_probe)

    def handle_all_area(self):
        """计算各停留区内的sn，duration，d"""
        for (date, tr) in self.tr_dict.items():
            cluster_set = set()
            for point in tr.plist:
                cluster_set.add(point.sr)
                if point.sr not in self.area_dict:
                    self.area_dict[point.sr] = StopoverArea()
                self.area_dict[point.sr].duration += point.duration
                self.area_dict[point.sr].d += point.total_data
                self.area_dict[point.sr].coordinate = [point.longitude, point.latitude]

            for cluster_id in cluster_set:
                self.area_dict[cluster_id].sn += 1
        for (cluster_id, area) in self.area_dict.items():
            area.d /= self.n
            self.sum_duration += area.duration
            self.sum_d += area.d

    def sort_in_des_order(self):
        """根据停留区域的duration进行降序排列"""
        result = sorted(self.area_dict.items(), key=lambda kv: (kv[1], kv[0]))
        # print(result)
        self.area_dict = dict(result)

    def output(self):
        print('=' * 40)
        print("Stopover Area Information")
        print('=' * 40)
        for (idx, area) in self.area_dict.items():
            print(idx)
            print('=' * 20)
            print("sn: %s duration: %s d: %s" % (area.sn, area.duration, area.d))
