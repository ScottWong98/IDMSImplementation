import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class POI:
    """Baidu POI information"""

    def __init__(self, name, category, location, coordinate):
        # POI name
        self.name = name
        # multi-level: list
        self.category = category
        # （省，市，区）
        self.location = location
        # (longitude, latitude)
        self.coordinate = coordinate

    def __str__(self):
        return f"{self.name}\t{self.category}\t{self.location}\t{self.coordinate}"


class POILoad:

    def __init__(self, filename):
        self.__filename = filename
        self.poi_list = []
        self.__data_x = []
        self.__data_y = []
        self.knn_model = None

    def load_poi(self):
        with open(self.__filename, 'r', encoding='UTF-8') as f:
            f_csv = csv.DictReader(f)
            items = list(f_csv)
            for item in items:
                # print(self.get_single_poi_from_line(item))
                poi_info = self.get_single_poi_from_line(item)
                self.__data_x.append(poi_info.coordinate)
                self.__data_y.append(poi_info.category)
                self.poi_list.append(poi_info)

    def train_data(self, k):
        self.load_poi()
        data_x = np.array(self.__data_x)
        data_y = np.array(self.__data_y)

        self.knn_model = KNeighborsClassifier(n_neighbors=k)
        self.knn_model.fit(data_x, data_y)

    @classmethod
    def get_single_poi_from_line(cls, item):
        return POI(name=item['名称'],
                   category=[item['大类'], item['中类'], item['小类']],
                   location=[item['省'], item['市'], item['区']],
                   coordinate=[float(item['lon']), float(item['lat'])])
