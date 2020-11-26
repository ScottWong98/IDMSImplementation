from .stop_area_mining import StopAreaMining
from .semantic_tag_conversion import SemanticTagConversion
from typing import Dict, List
import numpy as np
import pandas as pd


class TrajectoryGenerator:

    def __init__(self):
        self.sam: StopAreaMining = StopAreaMining()
        self.stc: SemanticTagConversion = None
        self.df: pd.DataFrame = None

    def stop_area_mining(
        self,
        filename: str,
        usecols: List[str],
        name_mapper: Dict[str, str],
        nan_dur_theta: float,
        dist_theta: float,
        point_dur_theta: float,
        eps: float,
        min_dur: float
    ) -> None:
        self.sam.load_data(filename=filename)
        self.sam.format_raw_data(usecols=usecols, name_mapper=name_mapper)
        self.sam.check_invalid_tr(nan_dur_theta=nan_dur_theta,
                                  dist_theta=dist_theta)
        self.sam.gen_valid_area(point_dur_theta=point_dur_theta,
                                eps=eps, min_dur=min_dur)
        self.sam.merge_adjacent_points()
        self.df = self.sam.df

    def semantic_tag_conversion(self, theta: float, poi_gen):
        self.stc = SemanticTagConversion(self.df)
        self.stc.main_area_mining(theta=theta)
        self.stc.semantic_tag_conversion(poi_gen=poi_gen)

    ######################################
    # The following method is to debug
    ######################################

    def test_load_data(self, filename: str) -> pd.DataFrame:

        self.sam.load_data(filename=filename)

        return self.sam.df

    def test_format_raw_data(self,
                             usecols: List[str],
                             name_mapper: Dict[str, str]) -> pd.DataFrame:
        self.sam.format_raw_data(usecols=usecols, name_mapper=name_mapper)
        return self.sam.df

    def test_check_invalid_data(self,
                                filename: str,
                                usecols: List[str],
                                name_mapper: Dict[str, str],
                                nan_dur_theta: float,
                                dist_theta: float) -> pd.DataFrame:
        sam = StopAreaMining()
        sam.load_data(filename=filename)
        sam.format_raw_data(usecols=usecols, name_mapper=name_mapper)
        sam.check_invalid_tr(nan_dur_theta=nan_dur_theta,
                             dist_theta=dist_theta)
        return sam.df

    def test_gen_valid_area(self,
                            filename: str,
                            usecols: List[str],
                            name_mapper: Dict[str, str],
                            nan_dur_theta: float,
                            dist_theta: float,
                            point_dur_theta: float,
                            eps: float,
                            min_dur: float) -> pd.DataFrame:
        sam = StopAreaMining()
        sam.load_data(filename=filename)
        sam.format_raw_data(usecols=usecols, name_mapper=name_mapper)
        sam.check_invalid_tr(nan_dur_theta=nan_dur_theta,
                             dist_theta=dist_theta)
        sam.gen_valid_area(point_dur_theta=point_dur_theta,
                           eps=eps, min_dur=min_dur)
        return sam.df
