import csv
import datetime


class Point:
    """Track point information"""
    def __init__(self, user_id, time, coordinate,
                 duration, day_number, total_data):
        self.user_id = user_id
        # (time_in, time_out)
        self.time = time
        # (longitude, latitude)
        self.coordinate = coordinate
        self.duration = duration
        self.day_number = day_number
        self.total_data = total_data
        # cluster flag, default = 0
        self.cluster_flag = 0
        # multi-level semantic flag
        self.sr = []

    def __str__(self):
        return f"{self.user_id}\t{self.time}\t{self.coordinate}\t" \
               f"{self.duration}\t{self.day_number}\t{self.total_data}\t" \
               f"{self.cluster_flag}\t{self.sr}"


class DataLoad:
    """Load data from file"""

    DATETIME_FORMAT = "%Y%m%d%H%M%S"

    def load_data(self, filename):
        tr_dict = {}
        with open(filename, 'r', encoding='UTF-8') as f:
            f_csv = csv.DictReader(f)
            items = list(f_csv)
            for item in items:
                _list = self.extract_data(item)
                point = Point(user_id=_list[0], time=_list[1],
                              coordinate=_list[2], duration=_list[3],
                              day_number=_list[4], total_data=_list[5])
                date = self.get_date_in_string(point.time[0])
                user_id = point.user_id

                if user_id not in tr_dict:
                    tr_dict[user_id] = {}
                if date not in tr_dict[user_id]:
                    tr_dict[user_id][date] = []
                tr_dict[user_id][date].append(point)
        return tr_dict

    def extract_data(self, item):
        user_id = item['MSISDN']
        time_in = self.string2datetime(item['STIME'])
        time_out = self.string2datetime(item['END_TIME'])
        longitude = float(item['NUMBERITUDE'])
        latitude = float(item['LATITUDE'])
        duration = int(item['DURATION'])
        day_number = int(item['DAY_NUMBER'])

        dt = item['DATA_TOTAL']
        total_data = 0.0
        if dt != '':
            total_data = float(dt)

        return [user_id, (time_in, time_out), (longitude, latitude), duration, day_number, total_data]

    @classmethod
    def string2datetime(cls, str_datetime):
        return datetime.datetime.strptime(str_datetime, cls.DATETIME_FORMAT)

    @classmethod
    def get_date_in_string(cls, raw_datetime):
        """ 20200101165902 -> 20200101"""
        return raw_datetime.strftime(cls.DATETIME_FORMAT)[:8]


def tr_dict_output(tr_dict):
    for user_id, user in tr_dict.items():
        print('=' * 40)
        print("User ID:", user_id)
        for date, tr in user.items():
            print('-' * 30)
            print("Date:", date)
            for idx, point in enumerate(tr):
                print(point)