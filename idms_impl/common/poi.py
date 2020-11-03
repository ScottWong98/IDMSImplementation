import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


class POI:
    """百度POI信息"""

    def __init__(self, name, category, location, coordinate):
        # POI名称
        self.name = name
        # 多层语义
        self.category = category
        # （省，市，区）
        self.location = location
        # 经纬度
        self.coordinate = coordinate

    def __str__(self):
        return f"{self.name}\t{self.category}\t{self.location}\t{self.coordinate}"


class POILoad:
    """加载POI信息"""

    def __init__(self, filename):
        # POI文件名称
        self.__filename = filename
        # POI列表
        self.poi_list = []
        # POI坐标列表
        self.__data_x = []
        # POI标签列表
        self.__data_y = []
        # KNN模型
        self.knn_model = None

    def load_poi(self):
        """加载POI"""
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
        """利用KNN对POI进行训练

        :param k: 邻居数量
        """
        self.load_poi()
        data_x = np.array(self.__data_x)
        data_y = np.array(self.__data_y)

        self.knn_model = KNeighborsClassifier(n_neighbors=k)
        self.knn_model.fit(data_x, data_y)

    @classmethod
    def get_single_poi_from_line(cls, item):
        """从文件中的一行数据读取有用的POI信息"""
        return POI(name=item['名称'],
                   category=[item['大类'], item['中类'], item['小类']],
                   location=[item['省'], item['市'], item['区']],
                   coordinate=[float(item['lon']), float(item['lat'])])
