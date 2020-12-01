import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN
from typing import List, Dict, NamedTuple


#  """ Define Stop Area Structure """
SA = NamedTuple('SA', [('left', int), ('right', int),
                       ('dur_sum', int), ('cid', int)])


def get_dist(p1: np.ndarray, p2: np.ndarray) -> np.float64:
    """Get the distance of two coordinate using haversine

    - Each value's unit in p1 and p2 is angle, so it must be converted to
        radian before using haversine.
    - p1 and p2: [latitude, longitude]

    Args:
        p1 (np.ndarray): first point [lat, lon]
        p2 (np.ndarray): second point [lat, lon]

    Returns:
        np.float64: the distance of p1 and p2 in angle unit
    """
    p1 = p1 * math.pi / 180
    p2 = p2 * math.pi / 180
    d = haversine_distances([p1, p2])
    return d[0, 1] * 180 / math.pi


def gen_cluster_labels(coords: np.ndarray, durs: np.ndarray, eps: float, min_dur: int) -> np.ndarray:
    """Generate cluster labels by using DBSCAN

    Args:
        coords (np.ndarray): [[lat, lon], [lat, lon], ..., [lat,lon]]
        durs (np.ndarray): [dur1, dur2, ..., dur_n]
        eps (float): the radius in angle unit
        min_dur (int): the min_samples in DBSCAN

    Returns:
        np.ndarray: the labels of each point after clustering.
                    `-1` is invalid point
                    eg: [-1, 0, 1, 2, 3, 0, 0]
    """
    coords = coords * math.pi / 180
    reps = eps * math.pi / 180
    clustering = DBSCAN(eps=reps, min_samples=min_dur, algorithm='ball_tree', metric='haversine')\
        .fit(X=coords, sample_weight=durs)
    return clustering.labels_


class StopAreaMining:

    def __init__(self, df):
        # df is one user's data
        self.df = df
        self.core_coords = None

    def handle_invalid_tr(self, nan_dur_theta, dist_theta, optimize=True):

        nan_filt_in_df = np.isnan(self.df['LATITUDE']) | np.isnan(
            self.df['LONGITUDE']) | pd.isna(self.df['ZH_LABEL'])

        if not optimize:
            self.df.drop(self.df[nan_filt_in_df].index, inplace=True)
            return

        self.df = self.df.groupby(['STAT_DATE'], sort=False).apply(self.__check_invalid_tr,
                                                                   nan_dur_theta,
                                                                   dist_theta)

        self.df.reset_index(drop=True, inplace=True)

    def delete_invalid_points(self, point_dur_theta):

        self.df = self.df[self.df['DURATION'] > point_dur_theta]
        self.df.reset_index(drop=True, inplace=True)

    def gen_cluster(self, eps, min_dur):

        coords = self.df.loc[:, ['LATITUDE', 'LONGITUDE']].values
        durs = self.df.loc[:, 'DURATION'].values

        cluster_flags = gen_cluster_labels(coords, durs, eps, min_dur)

        self.df['CLUSTER_ID'] = cluster_flags.reshape(
            cluster_flags.shape[0], 1)
        self.df = self.df[self.df['CLUSTER_ID'] != -1]
        self.df.reset_index(drop=True, inplace=True)

    def delete_invalid_area(self, min_dur):

        self.df = self.df.groupby(['STAT_DATE'], sort=False).apply(
            self.__delete_invalid_area, min_dur)
        self.df.reset_index(drop=True, inplace=True)

    def gen_core_coords(self):
        self.core_coords = self.df.groupby(['CLUSTER_ID'], sort=False).apply(
            lambda cluster: cluster[['LATITUDE', 'LONGITUDE']].mean()
        )
        self.df.reset_index(drop=True, inplace=True)

    def merge_adjacent_points(self):
        self.df = self.df.groupby(['STAT_DATE'], sort=False).apply(
            self.__merge_adjacent_points
        )
        self.df.drop(['END_TIME', 'ZH_LABEL'], axis=1, inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    def __check_invalid_tr(self, tr, nan_dur_theta, dist_theta):
        nan_filt = np.isnan(tr['LATITUDE']) | np.isnan(
            tr['LONGITUDE']) | pd.isna(tr['ZH_LABEL'])

        nan_dur_sum = tr[nan_filt].DURATION.sum()

        if nan_dur_sum == 0:
            return tr

        if nan_dur_sum >= nan_dur_theta:
            return tr.drop(tr.index)

        nan_index = tr[nan_filt].index
        n = nan_index.shape[0]
        left = nan_index[0] - 1
        coord_col = ['LATITUDE', 'LONGITUDE']
        for i in range(1, n):
            if nan_index[i] - nan_index[i - 1] != 1:
                d = get_dist(self.df.loc[left, coord_col],
                             self.df.loc[nan_index[i - 1] + 1, coord_col])
                if d >= dist_theta:
                    return tr.drop(tr.index)
                left = nan_index[i] - 1
        d = get_dist(self.df.loc[left, coord_col],
                     self.df.loc[nan_index[-1] + 1, coord_col])
        if d >= dist_theta:
            return tr.drop(tr.index)

        return tr.drop(nan_index)

    def __delete_invalid_area(self, tr, min_dur):

        durs, cids = tr['DURATION'].values, tr['CLUSTER_ID'].values
        sas: List[SA] = []
        n, left = tr.shape[0], 0

        # generate SAs
        for i in range(1, n):
            if cids[i] != cids[i - 1]:
                sas.append(SA(left, i - 1, durs[left:i].sum(), cids[i - 1]))
                left = i
        sas.append(SA(left, n - 1, durs[left:-1].sum(), cids[-1]))

        is_valid = np.array([True] * tr.shape[0])
        n, left = len(sas), 0

        # find the first SA where dur_sum >= min_dur
        for i in range(n):
            if sas[i].dur_sum >= min_dur:
                left = i
                break
        is_valid[0:sas[left].left] = False

        for i in range(left + 1, n):
            if sas[i].cid == sas[left].cid or sas[i].dur_sum >= min_dur:
                left = i
            else:
                is_valid[sas[i].left:sas[i].right+1] = False
        tr = tr[is_valid]

        return tr

    def __merge_adjacent_points(self, tr):
        cid_diff = tr['CLUSTER_ID'].diff()
        cid_diff.fillna(value=1., inplace=True)
        cid_diff = cid_diff.astype(bool)

        area_index = cid_diff[cid_diff].index.values

        for i in range(area_index.shape[0]):
            left = area_index[i]
            if i + 1 == area_index.shape[0]:
                right = tr.index[-1] + 1
            else:
                right = area_index[i + 1]

            tr.loc[left, 'DURATION'] = self.df.loc[left:right - 1, 'DURATION'].sum()
            tr.loc[left, 'TOTAL_DATA'] = \
                self.df.loc[left:right - 1, 'TOTAL_DATA'].sum()
            tr.loc[left, ['LATITUDE', 'LONGITUDE']] = \
                self.core_coords.loc[tr.loc[left, 'CLUSTER_ID']]

        return tr.loc[area_index]
