import pandas as pd
import numpy as np
import math


class SemanticTagConversion:

    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df
        self.cluster_attr: pd.DataFrame = None

    def home_prob(self, cid, n) -> float:
        sn = self.cluster_attr.loc[cid, 'sn']
        sn_prob = sn / n
        d = self.cluster_attr.loc[cid, 'd']
        sum_d = self.cluster_attr.loc[cid[0], 'd'].sum()
        d_prob = 0. if sum_d == 0. else d / sum_d
        return sn_prob + (1 - d_prob)

    def change_cluster_type(self, sr_list, n) -> None:
        n_sr = len(sr_list)
        if n_sr == 1:
            self.cluster_attr.loc[sr_list[0], 'type'] = 'home'
        elif n_sr == 2:
            if self.home_prob(sr_list[0], n) > self.home_prob(sr_list[1], n):
                self.cluster_attr.loc[sr_list[0], 'type'] = 'home'
                self.cluster_attr.loc[sr_list[1], 'type'] = 'work'
            else:
                self.cluster_attr.loc[sr_list[0], 'type'] = 'work'
                self.cluster_attr.loc[sr_list[1], 'type'] = 'home'
        elif n_sr == 3:
            d0 = self.cluster_attr.loc[sr_list[0], 'd']
            d1 = self.cluster_attr.loc[sr_list[1], 'd']
            d2 = self.cluster_attr.loc[sr_list[2], 'd']
            if d0 > (d1 + d2) / 2:
                self.cluster_attr.loc[sr_list[0], 'type'] = 'work'
                self.cluster_attr.loc[sr_list[1], 'type'] = 'home'
                self.cluster_attr.loc[sr_list[2], 'type'] = 'home'
            else:
                self.cluster_attr.loc[sr_list[0], 'type'] = 'home'
                self.cluster_attr.loc[sr_list[1], 'type'] = 'work'
                self.cluster_attr.loc[sr_list[2], 'type'] = 'work'

    def main_area_mining(self, theta):
        def gen_cluster_attr(user):
            cluster_grp = user.groupby(['CLUSTER_ID'], sort=False)
            attr = cluster_grp.apply(lambda x: pd.Series([x['STAT_DATE'].nunique(), x['DURATION'].sum(),
                                                          x['TOTAL_DATA'].sum() / x['STAT_DATE'].nunique()],
                                                         index=['sn', 'dur_sum', 'd']))
            attr.sort_values('dur_sum', ascending=False, inplace=True)
            return attr

        def gen_main_area(user):
            n = user['STAT_DATE'].nunique()
            uid = user.iloc[0, 0]
            dur_sum_in_user = self.cluster_attr.loc[uid, 'dur_sum'].sum()

            cur_cluster_attr = self.cluster_attr.loc[uid]

            sr_list = []
            tmp_dur_sum = 0
            for cluster_id, row in cur_cluster_attr.T.iteritems():
                tmp_dur_sum += row['dur_sum']
                if tmp_dur_sum / dur_sum_in_user <= theta:
                    sr_list.append((uid, cluster_id))
            self.change_cluster_type(sr_list, n)

        user_grp = self.df.groupby(['USER_ID'], sort=False)
        self.cluster_attr = user_grp.apply(gen_cluster_attr)

        self.cluster_attr['type'] = np.array(
            ['other'] * self.cluster_attr.shape[0])

        user_grp.apply(gen_main_area)

    def semantic_tag_conversion(self):
        user_grp = self.df.groupby(['USER_ID'], sort=False)
        # return a dataframe
        # the index is user_id and cluster_id
        #
        user_grp.apply()

        # add poi
        #
