import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors


class POIGenerator:

    def __init__(self):
        self.poi_df: pd.DataFrame = None
        self.knn_model = None
        self.main_model = None

    def load_poi(self, filename):
        self.poi_df = pd.read_csv(filename, encoding='utf')
        usecols = ['大类', '中类', '小类', '名称', '省', '市', '区', 'lat', 'lon']
        self.poi_df = self.poi_df[usecols]
        self.poi_df.rename(columns={
            '大类': 'big_ctg',
            '中类': 'medium_ctg',
            '小类': 'small_ctg',
            '名称': 'name',
            '省': 'province',
            '市': 'city',
            '区': 'region',
            'lat': 'poi_lat',
            'lon': 'poi_lon',
        }, inplace=True)

    def train_data(self, k):
        data_x = self.poi_df.loc[:, ['poi_lat', 'poi_lon']].values
        self.knn_model = NearestNeighbors(
            n_neighbors=k, metric='haversine', algorithm='ball_tree'
        )
        self.knn_model.fit(data_x)
        self.main_model = NearestNeighbors(
            n_neighbors=200, metric='haversine', algorithm='ball_tree'
        )
        self.main_model.fit(data_x)
