import csv
import datetime


def string2datetime(_raw_datetime):
    """ Convert String to Datetime """
    return datetime.datetime.strptime(_raw_datetime, "%Y%m%d%H%M%S")


def load_data(filename):
    data = {}
    with open(filename, 'r', encoding='UTF-8') as f:
        f_csv = csv.DictReader(f)
        items = list(f_csv)
        for item in items:
            msisdn = item['MSISDN']
            time_in = string2datetime(item['STIME'])
            time_out = string2datetime(item['END_TIME'])
            longitude = float(item['NUMBERITUDE'])
            latitude = float(item['LATITUDE'])
            duration = int(item['DURATION'])
            _list = [time_in, time_out, longitude, latitude, duration]
            if msisdn in data:
                data[msisdn].append(_list)
            else:
                data[msisdn] = [_list]
    return data

# result = {'a': [1, 2]}
#
# result['a'].append(3)
# result['b'] = [2, 3]
# print(result)


if __name__ == '__main__':
    data_set = load_data('../resource/test_13_user_with_flag.csv')
    print(len(data_set))
    for msisdn, tr in data_set.items():
        print(msisdn)
        print(tr)