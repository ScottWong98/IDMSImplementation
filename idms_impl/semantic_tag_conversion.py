import pandas as pd
import numpy as np
import math
from typing import List
from collections import Counter


class SemanticTagConversion:

    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        # cluster's attributes(sn, dur_sum, d)
        self.cluster_attr: pd.DataFrame = None
        self.cluster_poi: pd.DataFrame = None

    def home_prob(self, cid: tuple, n: int) -> float:
        """Calculate home prob

        home_prob = sn_prob + (1 - d_prob)

        sn_prob = sn / n

        d_prob = d / sum(d)

        Args:
            cid (tuple): (<user id>, <cluster_id>)
            n (int): observation days of this user

        Returns:
            float: the result of home prob
        """
        sn = self.cluster_attr.loc[cid, 'sn']
        sn_prob = sn / n
        d = self.cluster_attr.loc[cid, 'd']
        sum_d = self.cluster_attr.loc[cid[0], 'd'].sum()
        # sum_d may be zone
        d_prob = 0. if sum_d == 0. else d / sum_d
        return sn_prob + (1 - d_prob)

    def check_cluster_type(self, sr_list: List, n: int) -> None:
        """ Check cluster type

        Check cluster type is `home`, `work` or `other`

        Args:
            sr_list (List): the cluster id 
            n (int): obversion days of this user
        """

        # initialize the `type` column
        self.cluster_attr['type'] = np.array(['other'] *
                                             self.cluster_attr.shape[0])
        n_sr = len(sr_list)
        if n_sr == 1:
            types = ['home']
        elif n_sr == 2:
            if self.home_prob(sr_list[0], n) > self.home_prob(sr_list[1], n):
                types = ['home', 'work']
            else:
                types = ['work', 'home']
        else:
            d = self.cluster_attr.loc[sr_list, 'd'].values
            if d[0] > (d[1] + d[2]) / 2:
                types = ['work', 'home', 'home']
            else:
                types = ['home', 'work', 'work']
        self.cluster_attr.loc[sr_list, 'type'] = types

    def main_area_mining(self, theta: float) -> None:
        """ Main Area Mining

        For each user, get the attributes of all clusters(eg: sn, dur_sum, d), and sort it
        by dur_sum in descending order.

        For each user, judge each cluster's type (home, work, other).

        Args:
            theta (float): use it to generate sr_list
        """
        def gen_cluster_attr(user):
            cluster_grp = user.groupby(['CLUSTER_ID'], sort=False)
            _clusters = cluster_grp.apply(lambda x: pd.Series([x['STAT_DATE'].nunique(), x['DURATION'].sum(),
                                                               x['TOTAL_DATA'].sum() / x['STAT_DATE'].nunique()],
                                                              index=['sn', 'dur_sum', 'd']))
            _clusters.sort_values('dur_sum', ascending=False, inplace=True)
            return _clusters

        def gen_main_area(user):
            # n is the obversion days
            n = user['STAT_DATE'].nunique()
            uid = user.iloc[0, 0]
            dur_sum_in_user = self.cluster_attr.loc[uid, 'dur_sum'].sum()

            # all clusters in this user
            cur_cluster_attr = self.cluster_attr.loc[uid]
            # sr_list: the index of cluster in self.cluster_attr
            sr_list, tmp_dur_sum = [], 0
            for cluster_id, row in cur_cluster_attr.T.iteritems():
                tmp_dur_sum += row['dur_sum']
                if tmp_dur_sum / dur_sum_in_user <= theta:
                    sr_list.append((uid, cluster_id))
            self.check_cluster_type(sr_list, n)
            return self.cluster_attr.loc[uid]

        user_grp = self.df.groupby(['USER_ID'], sort=False)
        self.cluster_attr = user_grp.apply(gen_cluster_attr)

        self.cluster_attr = user_grp.apply(gen_main_area)
        # print(self.cluster_attr)

    def semantic_tag_conversion(self, poi_gen):

        def handle_each_cluster(cluster: pd.Series):
            def get_ctg(base_ctg):
                for ctg in ctgs:
                    if ctg[0] == base_ctg:
                        return ctg

            def find_poi():
                if cluster_type == 'home':
                    for base_ctg, _ in base_ctg_counter:
                        if base_ctg in home_list:
                            return get_ctg(base_ctg)
                elif cluster_type == 'work':
                    for base_ctg, _ in base_ctg_counter:
                        if base_ctg in work_list:
                            return get_ctg(base_ctg)
                return get_ctg(base_ctg_counter[0][0])
            coords = cluster[:2].values.reshape(1, 2)
            cluster_type = cluster[-1]
            poi_index = poi_gen.knn_model.kneighbors(
                coords, return_distance=False).flatten()
            ctgs = poi_gen.poi_df.iloc[poi_index, :7].values
            base_ctg_counter = Counter(ctgs[:, 0]).most_common()
            poi = find_poi()
            poi_series = pd.Series(poi, index=['big_ctg', 'medium_ctg', 'small_ctg',
                                               'name', 'province', 'city', 'region'])
            cluster = cluster.append(poi_series)
            return cluster

        home_list = ['商务住宅', '住宿服务', '生活服务', '地名地址信息']
        work_list = ['公司企业']

        # Generate cluster poi
        c_grp = self.df.groupby(['USER_ID', 'CLUSTER_ID'], sort=False)
        self.cluster_poi = c_grp.head(1)[['USER_ID', 'CLUSTER_ID',
                                          'LATITUDE', 'LONGITUDE']]
        self.cluster_poi.set_index(['USER_ID', 'CLUSTER_ID'], inplace=True)
        self.cluster_poi['type'] = self.cluster_attr['type']
        self.cluster_poi = self.cluster_poi.apply(handle_each_cluster, axis=1)

        self.df.set_index(['USER_ID', 'CLUSTER_ID'], inplace=True)
        self.df[['type', 'big_ctg', 'medium_ctg', 'small_ctg',
                 'name', 'province', 'city', 'region']] = self.cluster_poi.iloc[:, 2:]
        self.df.reset_index(drop=False, inplace=True)
