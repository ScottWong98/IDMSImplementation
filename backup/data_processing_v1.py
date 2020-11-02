import csv
import datetime
import time
import random
import numpy as np
from sklearn.neighbors import KDTree

start = time.process_time()


class VisitList:

    def __init__(self, num=0):
        self.unvisited_list = [i for i in range(num)]
        self.unvisited_num = num
        self.visited_list = list()

    def visit(self, point_id):
        self.visited_list.append(point_id)
        self.unvisited_list.remove(point_id)
        self.unvisited_num -= 1


def string2datetime(_raw_datetime):
    """ Convert String to Datetime """
    return datetime.datetime.strptime(_raw_datetime, "%Y%m%d%H%M%S")


def load_data(filename, ):

    data = []
    data_loc = []
    with open(filename, 'r', encoding='UTF-8') as f:
        f_csv = csv.DictReader(f)
        items = list(f_csv)

        for item in items:
            time_in = string2datetime(item['STIME'])
            time_out = string2datetime(item['END_TIME'])
            longitude = float(item['NUMBERITUDE'])
            latitude = float(item['LATITUDE'])
            duration = int(item['DURATION'])
            data.append([time_in,
                         time_out,
                         longitude,
                         latitude,
                         duration])
            data_loc.append([longitude, latitude])
            #print("TimeIn: %s, TimeOut: %s, Longitude: %s, Latitude:%s, Dura: %s"
            #      %(time_in, time_out, longitude, latitude, dura))
    return np.array(data), np.array(data_loc)


def calculate_duration_sum(data, neighbor_points):
    duration_sum = 0
    for i in neighbor_points:
        duration_sum += data[i][4]
    return duration_sum


def generate_cluster(data, data_loc, eps, min_duration):

    points_number = data_loc.shape[0]
    visit_list = VisitList(num=points_number)

    flag = -1
    cluster_list = [-1 for i in range(points_number)]

    tree = KDTree(data_loc)


    """
    current_point_id = random.choice(visit_list.unvisited_list)
    # print(data_loc)
    # print(current_point_id)
    # print(data_loc[current_point_id])
    neighbor_points = tree.query_radius(np.array([data_loc[current_point_id]]), r=eps)
    duration_sum = calculate_duration_sum(data, neighbor_points)
    print(duration_sum)
    """

    while visit_list.unvisited_num > 0:

        # Generate the current point's id from unvisited list randomly
        cur_point_id = random.choice(visit_list.unvisited_list)
        #print("===> Debug <== cur_point_id: %d" % cur_point_id)

        visit_list.visit(cur_point_id)

        neighbor_points = tree.query_radius(np.array([data_loc[cur_point_id]]), eps)[0].tolist()
        #print(type(neighbor_points))
        #neighbor_points.append(-19)
        #print(neighbor_points[-1])
        #break
        duration_sum = calculate_duration_sum(data, neighbor_points)
        #print(duration_sum)
        # print("===> Debug <== The Size of neighbor points: %d" % len(neighbor_points))
        if duration_sum >= min_duration:
            flag += 1
            cluster_list[cur_point_id] = flag
            for point in neighbor_points:
                # print("===> Debug <== The Size of neighbor points: %d" % len(neighbor_points))
                #print("===> Debug <== point: %d" % point)
                if point in visit_list.unvisited_list:
                    # print("===> Debug <== The current point is unvisited")
                    visit_list.visit(point)
                    npn_points = tree.query_radius(np.array([data_loc[point]]), eps)[0].tolist()
                    # print(len(npn_points))
                    if calculate_duration_sum(data, npn_points) >= min_duration:
                        #print(calculate_duration_sum(data, npn_points))
                        for i in npn_points:
                            if i not in neighbor_points:
                                # print("==> DEBUG <== Add one element to neighbor_points")
                                neighbor_points.append(i)
                if cluster_list[point] == -1:
                    cluster_list[point] = flag
            # print("*******************")
        else:
            cluster_list[cur_point_id] = -1

    return cluster_list


def kd_tree():
    rng = np.random.RandomState(0)
    X = rng.random_sample((10, 2))
    print(X)
    print(X[0])
    tree = KDTree(X, leaf_size=2)
    print(tree.query_radius(np.array([X[0]]), r=0.3))
    print(X[:1])


if __name__ == '__main__':
    data_set, data_set_loc = load_data('../resource/test_13_user_with_flag.csv')
    #print(dataSet)
    #kd_tree()
    mylist = generate_cluster(data_set, data_set_loc, 0.0000005, 900000)
    cnt = 0
    maxx = -2
    for i in mylist:
        if i > maxx:
            maxx = i
        if i != -1:
            cnt += 1
    print(cnt)
    print(maxx)
    print(mylist)
    end = time.process_time()
    print("Running time: %s Seconds" %(end - start))

