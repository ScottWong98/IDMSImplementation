import numpy as np
import pandas as pd
from .stop_area_mining import StopAreaMining
from .semantic_tag_conversion import SemanticTagConversion
from typing import List, Dict
import os


class DataProcess:

    def __init__(
        self,
        filename: str,
        usecols: List[str],
        col_name_mapper: Dict[str, str],
        nan_dur_sum: int,
        dist_theta: float,
        point_dur_theta: int,
        eps: float,
        min_dur: int,
        poi_gen,
        dur_sum_theta: int,
    ):
        self.df: pd.DataFrame = None

        self.filename = filename
        self.usecols: List[str] = usecols
        self.col_name_mapper: Dict[str, str] = col_name_mapper
        self.nan_dur_sum: int = nan_dur_sum
        self.dist_theta: float = dist_theta
        self.point_dur_theta: int = point_dur_theta
        self.eps: float = eps
        self.min_dur: int = min_dur
        self.poi_gen = poi_gen
        self.dur_sum_theta: int = dur_sum_theta

    def load_data(self):

        self.df = pd.read_csv(self.filename, encoding='gbk')
        self.df = self.df[self.usecols]
        self.df.rename(columns=self.col_name_mapper, inplace=True)
        self.df.sort_values(by=['USER_ID', 'STIME'], inplace=True)
        self.df['TOTAL_DATA'] = self.df['TOTAL_DATA'].fillna(0.)

        filt = (self.df.DURATION == 0)
        self.df.drop(self.df[filt].index, inplace=True)

        self.df.reset_index(drop=True, inplace=True)

    def process(self):

        user_grp = self.df.groupby(['USER_ID'], sort=False)

        _df = None
        for _, user in user_grp:
            tr_gen = self.process_single_user(user)
            if tr_gen.df.shape[0] == 0 or tr_gen.norm_df is None:
                continue
            norm_df = tr_gen.norm_df
            _df = norm_df if _df is None else pd.concat([_df, norm_df])
        self.df = _df
        self.df.reset_index(drop=True, inplace=True)

    def process_single_user(self, user):

        tr_gen = TrajectoryGenerator(user)

        tr_gen.run(self.nan_dur_sum, self.dist_theta,
                   self.point_dur_theta, self.eps, self.min_dur,
                   self.poi_gen, self.dur_sum_theta)

        return tr_gen

    def output(self, dir_name):

        self.df.to_csv(os.path.join(dir_name, 'sem_tr.csv'),
                       encoding='gbk', index=False)


class TrajectoryGenerator:

    def __init__(self, df):

        self.df: pd.DataFrame = df
        self.norm_df: pd.DataFrame = None

    def run(
        self,
        nan_dur_sum,
        dist_theta,
        point_dur_theta,
        eps,
        min_dur,
        poi_gen,
        theta
    ):
        self.stop_area_mining(nan_dur_sum, dist_theta,
                              point_dur_theta, eps, min_dur)
        if self.df.shape[0] == 0:
            return
        self.semantic_tag_conversion(poi_gen, theta)
        self.normalize_trajectory()

    def stop_area_mining(self, nan_dur_sum, dist_theta, point_dur_theta, eps, min_dur):

        assert self.df is not None

        valid_columns = ['USER_ID', 'STAT_DATE', 'STIME', 'END_TIME',
                         'ZH_LABEL', 'LATITUDE', 'LONGITUDE', 'DURATION', 'TOTAL_DATA']
        assert list(self.df.columns) == valid_columns

        self.df.reset_index(drop=True, inplace=True)

        sam = StopAreaMining(self.df, nan_dur_sum, dist_theta,
                             point_dur_theta, eps, min_dur)
        sam.run()
        self.df = sam.df

    def semantic_tag_conversion(self, poi_gen, theta):

        assert self.df is not None

        valid_columns = ['USER_ID', 'STAT_DATE', 'STIME',
                         'LATITUDE', 'LONGITUDE', 'DURATION', 'TOTAL_DATA', 'CLUSTER_ID']
        assert list(self.df.columns) == valid_columns

        self.df.reset_index(drop=True, inplace=True)

        stc = SemanticTagConversion(self.df, poi_gen, theta)
        stc.run()

        self.df = stc.df

    def normalize_trajectory(self):

        columns = ['USER_ID', 'STAT_DATE', 'STIME', 'LATITUDE',
                   'LONGITUDE', 'small_ctg', 'medium_ctg', 'big_ctg']

        self.norm_df = self.df.loc[:, columns]

        self.norm_df.rename(columns={
            'small_ctg': 'SEM1',
            'medium_ctg': 'SEM2',
            'big_ctg': 'SEM3'}, inplace=True)
