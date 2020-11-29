import pandas as pd
import numpy as np
import math
from typing import List, Dict, NamedTuple
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN

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
    def __init__(self) -> None:
        self.df: pd.DataFrame = None

    def load_data(self, filename: str) -> None:
        self.df = pd.read_csv(filename, encoding='gbk')

    def format_raw_data(self, usecols: List[str], name_mapper: Dict[str, str]) -> None:
        """Format Raw Data

        - Extract useful columns
        - Change columns' name and order
        - Sort data by `USER_ID` and `STIME`
        - Set `nan` in `TOTAL_DATA` to `0`
        - Drop rows whose `DURATION` is `0`

        Args:
            usecols (List[str]): what columns and what order to use
            name_mapper (Dict): old_column_name: new_column_name
        """

        self.df = self.df[usecols]
        self.df.rename(columns=name_mapper, inplace=True)
        self.df.sort_values(by=['USER_ID', 'STIME'], inplace=True)
        self.df['TOTAL_DATA'] = self.df['TOTAL_DATA'].fillna(0.)

        filt = (self.df.DURATION == 0)
        self.df.drop(self.df[filt].index, inplace=True)

        self.df.reset_index(drop=True, inplace=True)

    def check_invalid_tr(self, nan_dur_theta: float, dist_theta: float) -> None:
        """Check Invalid Trajectory

        For each trajectory, find all the bad rows where coordinate is nan and calculate
        the sum of duration in the bad rows.

        Check the sum of duraiton:
            if zero, no action, just return the original trajectory
            else if greater than `nan_dur_theta`, delete this trajectory
            else if less than `nan_dur_theta`, 
                check the distance between the two ends of each nan trajectory:
                    if all of distances less than dist_theta, just delete the bad
                        rows in this trajectory
                    else, delete this trajectory

        Args:
            nan_dur_theta (float): judge if delete the trajectory
            dist_theta (float): judge if delete the trajectory 
        """

        def handle_single_tr(tr: pd.DataFrame) -> pd.DataFrame:
            nan_filt = np.isnan(tr['LATITUDE']) | np.isnan(tr['LONGITUDE']) |\
                pd.isna(tr['ZH_LABEL'])
            invalid_points_dur_sum = tr[nan_filt].DURATION.sum()

            if invalid_points_dur_sum == 0:
                return tr

            if invalid_points_dur_sum >= nan_dur_theta:
                return tr.drop(tr.index)

            nan_index = tr[nan_filt].index
            n = nan_index.shape[0]
            left = nan_index[0] - 1
            coord_col = ['LATITUDE', 'LONGITUDE']
            for i in range(1, n):
                if nan_index[i] - nan_index[i - 1] != 1:
                    d = get_dist(self.df.loc[left, coord_col],
                                 self.df.loc[nan_index[i - 1] + 1, coord_col])
                    if d > dist_theta:
                        return tr.drop(tr.index)
                    left = nan_index[i] - 1
            d = get_dist(self.df.loc[left, coord_col],
                         self.df.loc[nan_index[-1] + 1, coord_col])
            if d > dist_theta:
                return tr.drop(tr.index)

            return tr.drop(nan_index)

        tr_grp = self.df.groupby(['USER_ID', 'STAT_DATE'], sort=False)
        self.df = tr_grp.apply(handle_single_tr)
        self.df.reset_index(drop=True, inplace=True)

    def gen_valid_area(self, point_dur_theta: float, eps: float, min_dur: int) -> None:
        """Generate valid area

        Before clustering, you need to delete those points whose duration less than `point_dur_theta`
        (eg: 20s).

        Use the global function of `gen_cluster_labels` to cluster, and get the labels of all points.

        Insert `CLUSTER_ID` column in the end of DataFrame and delete those invalid points whose cluster id
        is `-1`.

        Delete the invalid stop area in each trajectory.

        Args:
            point_dur_theta (float): delete the point whose duration less than 
                                    point_dur_theta before clustering, in second unit
            eps (float): the eps of DBSCAN, its unit is angle
            min_dur (int): the minimun duration of one valid stop area
        """
        def delete_invalid_area(tr: pd.DataFrame) -> pd.DataFrame:
            durs, cids = tr['DURATION'].values, tr['CLUSTER_ID'].values
            sas: List[SA] = []
            n, left = tr.shape[0], 0

            for i in range(1, n):
                if cids[i] != cids[i - 1]:
                    sas.append(
                        SA(left, i - 1, durs[left:i].sum(), cids[i - 1]))
                    left = i
            sas.append(SA(left, n - 1, durs[left:-1].sum(), cids[-1]))

            is_valid = np.array([True] * tr.shape[0])
            n, left = len(sas), 0
            for i in range(n):
                if sas[i].dur_sum >= min_dur:
                    left = i
                    break
            is_valid[0:sas[left].left] = False
            for i in range(left + 1, n):
                if sas[i].cid == sas[left].cid or sas[i].dur_sum >= min_dur:
                    left = i
                else:
                    is_valid[sas[i].left:sas[i].right + 1] = False
            tr = tr[is_valid]

            return tr

        def handle_single_user(user: pd.DataFrame) -> pd.DataFrame:
            coords = user.loc[:, ['LATITUDE', 'LONGITUDE']].values
            durs = user.loc[:, 'DURATION'].values

            cluster_flags = gen_cluster_labels(coords=coords, durs=durs,
                                               eps=eps, min_dur=min_dur)
            user['CLUSTER_ID'] = cluster_flags.reshape(
                cluster_flags.shape[0], 1)
            user = user[user.CLUSTER_ID != -1]

            #  NOTE: Uncomment it if you want display the DataFrame which just doesn't
            #           contain the points whose cluster id is -1
            # user.reset_index(drop=True, inplace=True)

            return user

        self.df = self.df[self.df.DURATION > point_dur_theta]
        self.df.reset_index(drop=True, inplace=True)

        user_grp = self.df.groupby(['USER_ID'], sort=False)
        self.df = user_grp.apply(handle_single_user)
        self.df.reset_index(drop=True, inplace=True)

        # TODO: need to log not print
        # print("[INFO] Finish clustering...")

        tr_grp = self.df.groupby(['USER_ID', 'STAT_DATE'], sort=False)
        self.df = tr_grp.apply(delete_invalid_area)
        self.df.reset_index(drop=True, inplace=True)

    def merge_adjacent_points(self) -> None:
        """Merge Adjacent points in each trajectory

        - Calculate the core coordinate in every cluster for each user
        - In each Trajectory, merge the adjacent points to one stop area where
            the `duration` and 'total_data' is the sum of those adjacent points
        - Change each stop area's coordinate to core coordinate
        """

        def handle_each_user(user):
            cluster_grp = user.groupby(['CLUSTER_ID'], sort=False)
            core_coords = cluster_grp.apply(
                lambda x: x[['LATITUDE', 'LONGITUDE']].median())
            return core_coords

        def handle_each_tr(tr):
            c_diff = tr['CLUSTER_ID'].diff()
            c_diff.fillna(value=1., inplace=True)
            c_diff = c_diff.astype(bool)

            area_index: np.ndarray = c_diff[c_diff].index.values

            for i in range(area_index.shape[0]):
                left = area_index[i]
                right = tr.index[-1] + 1\
                    if i + 1 == area_index.shape[0] \
                    else area_index[i + 1]
                tr.loc[left, 'DURATION'] = \
                    self.df.loc[left:right - 1, 'DURATION'].sum()
                tr.loc[left, 'TOTAL_DATA'] = \
                    self.df.loc[left:right - 1, 'TOTAL_DATA'].sum()

            new_tr = tr.loc[area_index]

            return new_tr

        # calculate the core coordinate
        user_grp = self.df.groupby(['USER_ID'], sort=False)
        core_coords = user_grp.apply(handle_each_user)

        # merge adjacent points to one stop area
        tr_grp = self.df.groupby(['USER_ID', 'STAT_DATE'], sort=False)
        self.df = tr_grp.apply(handle_each_tr)

        # change each area's coordinate to core coordinate
        self.df.set_index(['USER_ID', 'CLUSTER_ID'], inplace=True)
        self.df.loc[core_coords.index, ['LATITUDE', 'LONGITUDE']] = core_coords

        # reset the index of the dataframe and its order
        self.df.reset_index(inplace=True)
        cols = self.df.columns.tolist()

        # ['USER_ID', 'CLUSTER_ID', 'STAT_DATE', 'STIME', 'END_TIME', 'ZH_LABEL', 'LATITUDE',
        # 'LONGITUDE', 'DURATION', 'TOTAL_DATA']
        # =>
        # ['USER_ID', 'STAT_DATE', 'STIME', 'END_TIME', 'LATITUDE','LONGITUDE',
        #  'DURATION', 'TOTAL_DATA', 'CLUSTER_ID']

        # delete 'ZH_LABEL' and move 'CLUSTER_ID' to the end
        cols = cols[:1] + cols[2:5] + cols[6:] + cols[1:2]
        self.df = self.df[cols]
