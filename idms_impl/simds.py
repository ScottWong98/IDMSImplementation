import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import haversine_distances
import math


def simds(query_tr_df, similar_tr_df, beta):

    query_coords = query_tr_df[['LATITUDE', 'LONGITUDE']].values
    similar_coords = similar_tr_df[['LATITUDE', 'LONGITUDE']].values
    sp_sml_value, h_meter = space_similarity(query_coords, similar_coords)

    query_sems = query_tr_df[['SEM1', 'SEM2', 'SEM3']].values.T
    similar_sems = similar_tr_df[['SEM1', 'SEM2', 'SEM3']].values.T
    se_sml_value, common_sems = sem_similarity(query_sems, similar_sems)

    similar_value = beta * sp_sml_value + (1 - beta) * se_sml_value

    return {
        'USER_ID': similar_tr_df.iloc[0, 0],
        'STAT_DATE': similar_tr_df.iloc[0, 1],
        'H_METER': h_meter,
        'SEM1': common_sems[0],
        'SEM2': common_sems[1],
        'SEM3': common_sems[2],
        'SP_SIM': sp_sml_value,
        'SEM_SIM': se_sml_value,
        'SIM_VALUE': similar_value
    }


def space_similarity(query_coords, similar_coords):

    query_coords *= math.pi / 180
    similar_coords *= math.pi / 180
    result = haversine_distances(query_coords, similar_coords)
    h1 = result.min(axis=1).max()
    h2 = result.min(axis=0).max()
    d_max = result.max()

    d = 1 - max(h1, h2) / d_max
    h_meter = max(h1, h2) * math.pi / 180 * 6371000
    return d, h_meter


def sem_similarity(query_sems, similar_sems):
    min_shape = min(query_sems.shape[1], similar_sems.shape[1])
    max_shape = max(query_sems.shape[1], similar_sems.shape[1])

    sim = np.array([0.] * 4)
    sim[0] = min_shape / max_shape

    common_sems = []
    for i in range(3):
        common_num = lcss(query_sems[i, :], similar_sems[i, :])
        common_sems.append(common_num)
        sim[i + 1] = common_num / min_shape

    if sim[0] == 0 or sim[-1] == 0:
        return 0

    sum_w = (4 + 1) * 4 / 2
    result = 1 / sum_w * sim[0]
    for i in range(1, 4):
        result += (5 - i) / sum_w * sim[i]
    return result, common_sems


def lcss(sems1, sems2):
    if sems1.shape[0] == 0 or sems2.shape[0] == 0:
        return 0
    if sems1[0] == sems2[0]:
        return 1 + lcss(sems1[1:], sems2[1:])
    return max(lcss(sems1[1:], sems2), lcss(sems1, sems2[1:]))
