import csv
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from collections import Counter


class POI:

    def __init__(self, name, category, location, coordinate):
        # 名称
        self.name = name
        # 多层类别：（大类，中类，小类）
        self.category = category
        # 位置：（省，市，区）
        self.location = location
        # 坐标：（经度，纬度）
        self.coordinate = coordinate

    def __str__(self):
        return f"{self.name} {self.category} {self.location} {self.coordinate}"


class POILoad:

    def load_poi(self, filename):
        data, X, y = [], [], []
        with open(filename, 'r', encoding='UTF-8') as f:
            f_csv = csv.DictReader(f)
            items = list(f_csv)
            for item in items:
                # print(self.get_single_poi_from_line(item))
                poi_info = self.get_single_poi_from_line(item)
                X.append(poi_info.coordinate)
                y.append(poi_info.category)
                data.append(poi_info)
        return data, X, y

    @classmethod
    def get_single_poi_from_line(cls, item):
        return POI(name=item['名称'],
                   category=[item['大类'], item['中类'], item['小类']],
                   location=[item['省'], item['市'], item['区']],
                   coordinate=[float(item['lon']), float(item['lat'])])


class POILibrary:

    def __init__(self, tr_dict, area_dict, sr_dict):
        self.tr_dict = tr_dict
        self.area_dict = area_dict
        self.sr_dict = sr_dict
        self.poi_list = []
        # For KNN
        self.X = []
        # For KNN
        self.y = []

    def load_poi(self):
        data_load = POILoad()
        self.poi_list, self.X, self.y = data_load.load_poi("../resource/nj_poi.csv")

    def get_sementic_dict(self):
        sementic_dict = {}

        self.load_poi()
        # 1. KNN
        X = np.array(self.X)
        y = np.array(self.y)
        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X, y)

        for (cluster_id, area) in self.area_dict.items():
            result = neigh.kneighbors([area.coordinate])
            category_list = [self.poi_list[index].category for index in result[1][0]]
            category_list = np.array(category_list)
            # print(category_list[:, 0])
            category_dict = Counter(category_list[:, 0])

            for b_ctg in category_dict:
                if self.judge_main_area(cluster_id, b_ctg):
                    sementic_dict[cluster_id] = self.get_multi_category(b_ctg, category_list)
                    break
            # 说明对于家和工作这种地点，在k个邻居里面没有找到匹配的
            # 暂时设置为K个邻居中出现次数最多的那个
            if cluster_id not in sementic_dict:
                b_ctg = list(category_dict.items())[0][0]
                sementic_dict[cluster_id] = self.get_multi_category(b_ctg, category_list)

        return sementic_dict

    def judge_main_area(self, cluster_id, base_category):
        if cluster_id not in self.sr_dict:
            return True
        area_flag = self.sr_dict[cluster_id]
        # TODO: 怎么判断是否为家
        home_list = ['商务住宅', '住宿服务', '生活服务', '地名地址信息']
        # TODO: 怎么判断是否为公司
        work_list = ['公司企业']

        if area_flag == 'HOME':
            match_num = [i for i in home_list if base_category == i]
            if len(match_num) == 0:
                return False
            else:
                return True
        else:
            match_num = [i for i in work_list if base_category == i]
            if len(match_num) == 0:
                return False
            else:
                return True

    @classmethod
    def get_multi_category(cls, base_category, category_list):
        for c in category_list:
            if base_category == c[0]:
                return c.tolist()

if __name__ == '__main__':
    poi_load = POILoad()
    data, X, y = poi_load.load_poi("../resource/nj_poi.csv")
    category_set = set()
    for element in data:
        category_set.add(element.category[0])

    print(category_set)
