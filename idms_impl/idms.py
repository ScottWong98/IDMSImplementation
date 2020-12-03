import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import math
from typing import NamedTuple, List
from .index_tree import IndexTree, IndexTreeNode
from .simds import simds


class IDMS:

    VALID_COLUMNS = ['USER_ID', 'STAT_DATE', 'STIME', 'LATITUDE',
                     'LONGITUDE', 'SEM1', 'SEM2', 'SEM3']

    SEF_INDEX = ['SEM3', 'SEM2', 'SEM1', 'GID']
    SPF_INDEX = ['GID', 'SEM3', 'SEM2', 'SEM1']

    EXTRACT_COLUMNS = ['TID', 'GID', 'SEM1', 'SEM2', 'SEM3']

    # 南昌
    O_LAT, O_LON = 28.683016, 115.857963

    def __init__(self):
        self.df: pd.DataFram = None
        self.sef_df: pd.DataFrame = None
        self.spf_df: pd.DataFrame = None
        self.grid_size = None
        self.sef_tree = None
        self.spf_tree = None

    def build(self, raw_df, grid_size):

        raw_df.reset_index(drop=True, inplace=True)
        self.grid_size = grid_size
        self.df = self.rasterize(raw_df)

        self.sef_df = self.gen_multi_level_index(
            self.df.copy(), self.SEF_INDEX)
        self.spf_df = self.gen_multi_level_index(
            self.df.copy(), self.SPF_INDEX)

        ######################

        self.sef_tree = IndexTree(self.sef_df)
        self.sef_tree.build()
        self.spf_tree = IndexTree(self.spf_df)
        self.spf_tree.build()

    def query(self, query_df, beta):

        assert query_df is not None

        assert len(query_df.groupby(['USER_ID', 'STAT_DATE'], sort=False)) == 1

        query_df.reset_index(drop=True, inplace=True)
        query_df = self.rasterize(query_df)

        if beta < 0.5:
            query_f_df = self.gen_multi_level_index(
                query_df.copy(), self.SEF_INDEX)
        else:
            query_f_df = self.gen_multi_level_index(
                query_df.copy(), self.SPF_INDEX)

        query_tree = IndexTree(query_f_df)
        query_tree.build()

        if beta < 0.5:
            tids = self.sef_tree.query(query_tree)
        else:
            tids = self.spf_tree.query(query_tree)

        res_list = []
        for tid in tids:
            filt = (self.df['USER_ID'] == tid[0]) & (
                self.df['STAT_DATE'] == tid[1])
            similar_tr_df = self.df[filt].loc[:, self.VALID_COLUMNS]
            similar_tr_df.sort_values(by=['STIME'], inplace=True)
            sim_info = simds(query_df, similar_tr_df, beta)
            res_list.append(sim_info)
        sim_df = pd.DataFrame(res_list)

        sim_df.sort_values(by=['SIM_VALUE'], ascending=False, inplace=True)
        sim_df.reset_index(drop=True, inplace=True)

        return tids, sim_df

    def update(self, new_df):

        new_df = self.rasterize(new_df)

        new_sef_df = self.gen_multi_level_index(
            new_df.copy(), self.SEF_INDEX)
        new_spf_df = self.gen_multi_level_index(
            new_df.copy(), self.SPF_INDEX)

        new_sef_tree = IndexTree(new_sef_df)
        new_sef_tree.build()
        new_spf_tree = IndexTree(new_spf_df)
        new_spf_tree.build()

        self.sef_tree.update(new_sef_tree)
        self.spf_tree.update(new_spf_tree)

    def rasterize(self, raw_df):
        assert list(raw_df.columns) == self.VALID_COLUMNS

        def gid(lat, lon):
            x = (lat - self.O_LAT) // self.grid_size
            y = (lon - self.O_LON) // self.grid_size
            return f"{int(x)}-{int(y)}"
        gid_ses = raw_df.apply(
            lambda x: gid(x['LATITUDE'], x['LONGITUDE']), axis=1)
        raw_df = raw_df.assign(GID=gid_ses)
        return raw_df

    def gen_multi_level_index(self, f_df, index):

        tid_ses = f_df.apply(
            lambda x: (x['USER_ID'], x['STAT_DATE']), axis=1
        )
        f_df = f_df.assign(TID=tid_ses)
        f_df = f_df[self.EXTRACT_COLUMNS]
        f_df.set_index(index, inplace=True)
        f_df = f_df.sort_index()
        return f_df
