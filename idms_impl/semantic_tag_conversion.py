import pandas as pd
import numpy as np
from collections import Counter


class SemanticTagConversion:

    HOME_LIST = ['商务住宅']
    WORK_LIST = ['公司企业']

    def __init__(
        self,
        df,
        poi_gen,
        dur_sum_theta,
    ):
        self.df: pd.DataFrame = df
        self.df.reset_index(drop=True, inplace=True)

        self.poi_gen = poi_gen
        self.dur_sum_theta = dur_sum_theta

        self.cluster_attr: pd.DataFrame = None

    def run(self):
        self.main_area_mining()
        self.semantic_tag_conversion()

    def main_area_mining(self):
        mam = MainAreaMining(self.df)
        mam.gen_cluster_attr()
        mam.gen_cluster_type(self.dur_sum_theta)
        self.cluster_attr = mam.cluster_attr

    def semantic_tag_conversion(self):

        self.cluster_attr = self.cluster_attr.apply(
            self.__semantic_tag_conversion, axis=1
        )

        addon_columns = list(self.cluster_attr.columns)[5:]
        _columns = list(self.df.columns) + addon_columns

        self.df.set_index(['CLUSTER_ID'], inplace=True)
        self.df[addon_columns] = self.cluster_attr.iloc[:, 5:]
        self.df.reset_index(drop=False, inplace=True)
        self.df = self.df[_columns]
        self.df.reset_index(drop=True, inplace=True)

    def __semantic_tag_conversion(self, cluster: pd.Series):
        coords = cluster[['core_lat', 'core_lon']].values.reshape(1, 2)
        ctype = cluster['type']
        if ctype == 'other':
            poi_index = self.poi_gen.knn_model.kneighbors(
                coords, return_distance=False
            ).flatten()
        else:
            poi_index = self.poi_gen.main_model.kneighbors(
                coords, return_distance=False
            ).flatten()
        pois = self.poi_gen.poi_df.iloc[poi_index, :]

        big_ctg = Counter(pois.iloc[:, 0].values).most_common(1)[0][0]

        poi = pois[pois['big_ctg'] == big_ctg].iloc[0]
        if ctype != 'other':
            if ctype == 'home':
                judge_list = self.HOME_LIST
            else:
                judge_list = self.WORK_LIST
            for row in self.poi_gen.poi_df.loc[poi_index].itertuples():
                if row.big_ctg in judge_list:
                    poi = pd.Series(
                        [row.big_ctg, row.medium_ctg, row.small_ctg, row.name, row.province, row.city, row.region,
                            row.poi_lat, row.poi_lon],
                        index=pois.columns
                    )
                    break

        cluster = cluster.append(poi)
        return cluster


class MainAreaMining:

    def __init__(self, df):
        self.df: pd.DataFrame = df
        self.cluster_attr: pd.DataFrame = None
        self.n = self.df['STAT_DATE'].nunique()
        self.sum_d = 0

    def gen_cluster_attr(self):
        cluster_grp = self.df.groupby(['CLUSTER_ID'], sort=False)
        self.cluster_attr = cluster_grp.apply(
            lambda cluster: pd.Series(
                [cluster['STAT_DATE'].nunique(), cluster['DURATION'].sum(),
                 cluster['TOTAL_DATA'].sum() / cluster['STAT_DATE'].nunique(),
                 cluster['LATITUDE'].iloc[0], cluster['LONGITUDE'].iloc[0]],
                index=['sn', 'dur_sum', 'd', 'core_lat', 'core_lon']
            )
        )
        self.cluster_attr.sort_values('dur_sum', ascending=False, inplace=True)
        self.sum_d = self.cluster_attr['d'].sum()

    def gen_cluster_type(self, theta):

        total_dur_sum = self.df['DURATION'].sum()
        sr_list, tmp_dur_sum = [], 0
        for row in self.cluster_attr.itertuples():
            tmp_dur_sum += row.dur_sum
            if tmp_dur_sum / total_dur_sum <= theta:
                sr_list.append(row.Index)
            else:
                sr_list.append(row.Index)
                break

        self.cluster_attr['type'] = np.array(
            ['other'] * self.cluster_attr.shape[0])

        n_sr = len(sr_list)

        if n_sr > 3:
            return
        if n_sr == 1:
            types = ['home']
        elif n_sr == 2:
            if self.home_prob(sr_list[0]) > self.home_prob(sr_list[1]):
                types = ['home', 'work']
            else:
                types = ['work', 'home']
        elif n_sr == 3:
            d = self.cluster_attr.loc[sr_list, 'd'].values
            if d[0] > (d[1] + d[2]) / 2:
                types = ['work', 'home', 'home']
            else:
                types = ['home', 'work', 'work']

        self.cluster_attr.loc[sr_list, 'type'] = types

    def home_prob(self, cid):

        assert self.n != 0

        sn = self.cluster_attr.loc[cid, 'sn']
        sn_prob = sn / self.n
        d = self.cluster_attr.loc[cid, 'd']
        d_prob = 0. if self.sum_d == 0. else d / self.sum_d
        return sn_prob + (1 - d_prob)
