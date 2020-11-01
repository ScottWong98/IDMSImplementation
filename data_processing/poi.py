import csv
from sklearn.neighbors import KNeighborsClassifier
import numpy as np


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

    def __init__(self):
        self.poi_list = []
        # For KNN
        self.X = []
        # For KNN
        self.y = []

    def load_poi(self):
        data_load = POILoad()
        self.poi_list, self.X, self.y = data_load.load_poi("../resource/nj_poi.csv")

    def generate_knn_model(self):
        self.load_poi()
        X = np.array(self.X)
        y = np.array(self.y)

        neigh = KNeighborsClassifier(n_neighbors=5)
        neigh.fit(X, y)
        return neigh, self.poi_list


if __name__ == '__main__':
    poi_load = POILoad()
    data, X, y = poi_load.load_poi("../resource/nj_poi.csv")
    category_set = set()
    for element in data:
        category_set.add(element.category[0])

    print(category_set)
