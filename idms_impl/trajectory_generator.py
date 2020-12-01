import numpy as np
import pandas as pd
from .stop_area_mining import StopAreaMining
from .semantic_tag_conversion import SemanticTagConversion


class TrajectoryGenerator:

    def __init__(self):
        self.df: pd.DataFrame = None

    def load_data(self, filename, usecols, name_mapper, out_filename=None):
        self.df = pd.read_csv(filename, encoding='gbk')
        self.df = self.df[usecols]
        self.df.rename(columns=name_mapper, inplace=True)
        self.df.sort_values(by=['USER_ID', 'STIME'], inplace=True)
        self.df['TOTAL_DATA'] = self.df['TOTAL_DATA'].fillna(0.)

        filt = (self.df.DURATION == 0)
        self.df.drop(self.df[filt].index, inplace=True)

        self.df.reset_index(drop=True, inplace=True)

        if out_filename is not None:
            self.df.to_csv(out_filename, encoding='gbk', index=False)

    def stop_area_mining(self, nan_dur_sum, dist_theta, point_dur_theta, eps, min_dur, filename=None):

        if filename is not None:
            self.df = pd.read_csv(filename, encoding='gbk')
        valid_columns = ['USER_ID', 'STAT_DATE', 'STIME', 'END_TIME',
                         'ZH_LABEL', 'LATITUDE', 'LONGITUDE', 'DURATION', 'TOTAL_DATA']
        assert self.df is not None

        assert list(self.df.columns) == valid_columns

        user_grp = self.df.groupby(['USER_ID'], sort=False)

        self.df = user_grp.apply(
            self.__stop_area_mining, nan_dur_sum, dist_theta, point_dur_theta, eps, min_dur)
        self.df.reset_index(drop=True, inplace=True)

    def semantic_tag_conversion(self, poi_gen, theta, filename=None):
        if filename is not None:
            self.df = pd.read_csv(filename, encoding='gbk')

        assert self.df is not None
        valid_columns = ['USER_ID', 'STAT_DATE', 'STIME',
                         'LATITUDE', 'LONGITUDE', 'DURATION', 'TOTAL_DATA', 'CLUSTER_ID']

        assert list(self.df.columns) == valid_columns

        user_grp = self.df.groupby(['USER_ID'], sort=False)

        df = None
        for _, user in user_grp:
            _df = self.__semantic_tag_conversion(user, poi_gen, theta)
            if df is None:
                df = _df
            else:
                df = pd.concat([df, _df])
        self.df = df

    def __stop_area_mining(self, user, nan_dur_sum, dist_theta, point_dur_theta, eps, min_dur):

        sam = StopAreaMining(user)
        sam.handle_invalid_tr(nan_dur_sum, dist_theta)
        sam.delete_invalid_points(point_dur_theta)
        sam.gen_cluster(eps, min_dur)
        sam.delete_invalid_area(min_dur)
        sam.gen_core_coords()
        sam.merge_adjacent_points()

        return sam.df

    def __semantic_tag_conversion(self, user, poi_gen, theta):
        stc = SemanticTagConversion(user, poi_gen)
        stc.main_area_mining(theta)
        stc.semantic_tag_conversion()

        return stc.df
