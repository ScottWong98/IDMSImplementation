class DataProcess:
    """数据处理
    将未处理的轨迹表转换成轨迹点为停留区域，并且具有多层语义的轨迹表
    """
    def __init__(self, tr_dict):
        # 轨迹表
        self.tr_dict = tr_dict
        self.plist = self.gen_plist()

    def stopover_area_mining(self, eps, min_duration):
        """停留区域点挖掘"""
        for user_id, user in self.tr_dict.items():
            pass
        pass

    def semantic_tag_conversion(self, n, theta):
        """语义标签转化"""
        pass

    def gen_plist(self):

        pass
