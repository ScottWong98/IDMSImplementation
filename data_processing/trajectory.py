class Trajectory:

    def __init__(self, user_id, date):
        self.user_id = user_id
        self.date = date
        self.plist = []

    def sort_point(self):
        pass

    def convert2sr_tr(self, cluster_list):
        """将tr改为SRtr"""
        # 1. 轨迹中每个点的sr更新为聚类标签
        self.change_point_sr2cluster(cluster_list)

        # 2. 删除
        self.remove_invalid_points()

        # 3. 合并
        self.merge_adjacent_points()

    def change_point_sr2cluster(self, cluster_list):
        """轨迹中每个点的sr更新为聚类标签"""
        for idx, point in enumerate(self.plist):
            point.sr2cluster(cluster_list[idx])

    @classmethod
    def get_core_coordinate(cls, xy_list):
        _x, _y = 0.0, 0.0
        for xy in xy_list:
            _x += xy[0]
            _y += xy[1]
        _len = len(xy_list)
        return _x / _len, _y / _len

    def merge_adjacent_points(self):
        """合并相邻的重复元素"""
        tmp_list = []
        tmp_point = None
        xy_list = []
        for idx, point in enumerate(self.plist):
            if idx == 0 or point.sr != self.plist[idx - 1].sr:
                if idx != 0:
                    _lon, _lat = self.get_core_coordinate(xy_list)
                    tmp_point.longitude = _lon
                    tmp_point.latitude = _lat
                    # self.plist[idx - 1] = tmp_point
                    tmp_list.append(tmp_point)

                tmp_point = point
                # TODO: <lon, lat> OR <lat, lon>
                xy_list = [[point.longitude, point.latitude]]

            else:
                tmp_point.duration += point.duration
                tmp_point.total_data += point.total_data
                tmp_point.time_out = point.time_out
                xy_list.append([point.longitude, point.latitude])
                # self.plist.remove(point)
        _lon, _lat = self.get_core_coordinate(xy_list)
        tmp_point.longitude = _lon
        tmp_point.latitude = _lat
        #self.plist[-1] = tmp_point
        tmp_list.append(tmp_point)
        self.plist = tmp_list

    def remove_invalid_points(self):
        for p in self.plist:
            if p.sr == 0:
                self.plist.remove(p)

    def output(self):
        print('=' * 40)
        print("User ID:", self.user_id)
        print('-' * 30)
        print("Date:", self.date)
        for idx, point in enumerate(self.plist):
            print(point)
