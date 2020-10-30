class POI:

    def __init__(self, name, category, location, coordinate):
        # 名称
        self.name = name
        # 多层类别：（大类，中类，小类）
        self.category = category
        # 位置：（省，市，区）
        self.location = location
        # 坐标：（经度，纬度）
        self.coordinate = coordinate

    def __str__(self):
        return f"{self.name} {self.category} {self.location} {self.coordinate}"

class POILibrary:

    def __init__(self, tr_dict, area_dict, sr_dict):
        self.tr_dict = tr_dict
        self.area_dict = area_dict
        self.sr_dict = sr_dict

