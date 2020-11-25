import pandas as pd
import numpy as np
import math
from typing import List, Dict
from sklearn.metrics.pairwise import haversine_distances
from sklearn.cluster import DBSCAN

Coord = np.ndarray
float64 = np.float64


def get_dist(p1: Coord, p2: Coord) -> float64:
    """Get the distance of two coordinate using haversine

    * Each value's unit in p1 and p2 is angle, so it must be converted to
        radian before using haversine. 
    * p1 and p2: [latitude, longitude]

    Args:
        p1 (Coord): first point [lat, lon]
        p2 (Coord): second point [lat, lon] 

    Returns:
        float64: the distance of p1 and p2 in angle unit
    """
    p1 = p1 * math.pi / 180
    p2 = p2 * math.pi / 180
    d = haversine_distances([p1, p2])
    return d[0, 1] * 180 / math.pi


class DataLoad:

    def __init__(self) -> None:
        self.df: pd.DataFrame = None

    def load_data(self, filename: str) -> None:
        self.df = pd.read_csv(filename, encoding='gbk')

    def format_raw_data(self, usecols: List[str], name_mapper: Dict) -> None:
        """ Format Raw Data

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

    def check_invalid_tr(self, dur_threshold: float, dist_threshold: float) -> None:
        """Check Invalid Trajectory

        For each trajectory, find all the bad rows where coordinate is nan and calculate
        the sum of duration in the bad rows.
        - Check the sum of duraiton
            - zero, no action, just return the original trajectory
            - greater than `dur_threshold`, delete this trajectory
            - less than `dur_threshold`, 
                check the distance between the two ends of each nan trajectory
                    - if all of distances less than dist_threshold, just delete the bad
                        rows in this trajectory
                    - else, delete this trajectory

        Args:
            dur_threshold (float): [description]
            dist_threshold (float): [description]
        """

        def handle_single_tr(tr):
            nan_filt = np.isnan(tr['LATITUDE']) | np.isnan(tr['LONGITUDE']) |\
                pd.isna(tr['ZH_LABEL'])
            invalid_points_dur_sum = tr[nan_filt].DURATION.sum()

            if invalid_points_dur_sum == 0:
                return tr

            if invalid_points_dur_sum >= dur_threshold:
                return tr.drop(tr.index)

            nan_index = tr[nan_filt].index
            n = nan_index.shape[0]
            left = nan_index[0] - 1
            coord_col = ['LATITUDE', 'LONGITUDE']
            for i in range(1, n):
                if nan_index[i] - nan_index[i - 1] != 1:
                    d = get_dist(self.df.loc[left, coord_col],
                                 self.df.loc[nan_index[i - 1] + 1, coord_col])
                    if d > dist_threshold:
                        return tr.drop(tr.index)
                    left = nan_index[i] - 1
            d = get_dist(self.df.loc[left, coord_col],
                         self.df.loc[nan_index[-1] + 1, coord_col])
            if d > dist_threshold:
                return tr.drop(tr.index)

            return tr.drop(nan_index)

        tr_grp = self.df.groupby(['USER_ID', 'STAT_DATE'], sort=False)
        self.df = tr_grp.apply(handle_single_tr)
        self.df.reset_index(drop=True, inplace=True)

    def clustering(self, eps: float, min_dur: int) -> None:

        pass

    def stop_area_mining(self):
        pass

    def main_stop_area_mining(self):
        pass

    def semantic_tags_convertion(self):
        pass


class StopAreaMining:

    def __init__(self):
        pass
