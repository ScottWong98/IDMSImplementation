from idms_impl.common.index_tree import IndexTree


class DataStorage:

    def __init__(self, tr_dict):
        self.tr_dict = tr_dict

    def build_index_tree(self, origin_point, side_length):
        """构建索引树，生成空间优先索引树和语义优先索引树"""
        tree = IndexTree(self.tr_dict)
        tree.build(origin_point, side_length)
        return tree.spf_rt, tree.sef_rt

