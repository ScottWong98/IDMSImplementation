from typing import NamedTuple, Tuple
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


class POIGenerator:

    def __init__(self):
        self.poi_df: pd.DataFrame = None
        self.knn_model = None

    def load_poi(self, filename):
        self.poi_df = pd.read_csv(filename, encoding='utf')
        usecols = ['大类', '中类', '小类', '名称', '省', '市', '区', 'lon', 'lat']
        self.poi_df = self.poi_df[usecols]

    def train_data(self, k):
        data_x = self.poi_df.iloc[:, -2:].values
        data_x = data_x[:, [1, 0]]
        # print(data_x)
        self.knn_model = NearestNeighbors(
            n_neighbors=k, metric='haversine', algorithm='ball_tree')
        self.knn_model.fit(data_x)
