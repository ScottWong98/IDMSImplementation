import datetime
import csv
from data_processing.point import Point
from data_processing.user import User
from data_processing.trajectory import Trajectory


class DataLoad:
    """从文件中加载数据，提取中关键数据属性"""

    FORMAT_DATETIME = "%Y%m%d%H%M%S"

    def __init__(self, filename):
        self.filename = filename

    def load_data(self):
        """
        加载数据
        :return: 包含User的列表
        """
        data = {}
        with open(self.filename, 'r', encoding='UTF-8') as f:
            f_csv = csv.DictReader(f)
            items = list(f_csv)
            for item in items:
                _list = self.extract_data(item)
                point = Point(_list[0], _list[1], _list[2], _list[3], _list[4], _list[5], _list[6], _list[7])
                date_str = self.get_date_in_string(_list[1])
                user_id = point.user_id

                if user_id not in data:
                    data[user_id] = User(user_id)

                if date_str not in data[user_id].tr_dict:
                    data[user_id].tr_dict[date_str] = Trajectory(user_id, date_str)

                data[user_id].tr_dict[date_str].plist.append(point)
        # self.print_user_dict(data)
        return data

    @classmethod
    def string2datetime(cls, str_datetime):
        """string格式的日期转化成datetime格式"""
        return datetime.datetime.strptime(str_datetime, cls.FORMAT_DATETIME)

    def extract_data(self, item):
        """提取有用的数据属性
        :param item: csv文件中的一行记录
        :return:
            - 用户标识
            - 包含有用数据信息的列表
        """
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

        return [user_id, time_in, time_out, longitude, latitude, duration, day_number, total_data]

    @classmethod
    def get_date_in_string(cls, raw_datetime):
        """ 20200101165902 -> 20200101"""
        return raw_datetime.strftime(cls.FORMAT_DATETIME)[:8]

    @classmethod
    def print_user_dict(cls, user_dict):
        """ Print User Dict
        eg:
            ========================================
            User ID: 121399
            ------------------------------
            Date: 20200101
            121399	2020-01-01 00:58:43	2020-01-01 02:38:48	118.711894	31.965642	6005	1	11.2890625
            ...
            ------------------------------
            Date: 20200102
            121399	2020-01-02 00:47:52	2020-01-02 02:01:58	118.70964199999999	31.962601	4446	2	11.28515625
            ...
            ========================================
        """
        for (user_id, user) in user_dict.items():
            # ==> 121399 <data_processing.user.User object at 0x000001E937A5EA88> <==
            print('=' * 40)
            print("User ID:", user_id)
            for (date, tr) in user.tr_dict.items():
                print('-' * 30)
                print("Date:", date)

                for idx, point in enumerate(tr.plist):
                    print(point)


if __name__ == '__main__':
    data_load = DataLoad("../resource/test_input_2.csv")
    data = data_load.load_data()
    data_load.print_user_dict(data)
    print(data_load.load_data())

