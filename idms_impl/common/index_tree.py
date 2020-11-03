import math


class IndexTreeNode:
    """索引表结点"""
    def __init__(self, parent=None, children=None, name=None, depth=0, tr_id_set=None):
        self.parent = parent
        self.children = children
        self.name = name
        self.depth = depth
        self.tr_id_set = tr_id_set

    def sort_children(self):
        if len(self.children) == 0:
            return
        ordered_dict = {}
        if isinstance(list(self.children.keys())[0], tuple):
            sort_list = sorted(self.children.items(), key=lambda d: d[0])
        else:
            sort_list = sorted(self.children.items(), key=lambda d: d[0].encode('gbk'))

        for item in sort_list:
            ordered_dict[item[0]] = item[1]
        self.children = ordered_dict


class EdgeExtract:
    """从tr_dict预处理出所有的边"""

    def __init__(self, tr_dict, origin_point, side_length):
        self.tr_dict = tr_dict
        self.origin_point = origin_point
        self.side_length = side_length

    def extract(self):
        spf_edges, sef_edges = [], []
        for user_id, user_tr_dict in self.tr_dict.items():
            for date, tr in user_tr_dict.items():
                tr_id = (user_id, date)
                for idx, p in enumerate(tr):
                    # 栅格化过后的编码ID
                    code_id = self.__rasterize(p.coordinate)
                    spf_list, sef_list = ['RT', code_id], ['RT']
                    for element in p.sr:
                        spf_list.append(element)
                        sef_list.append(element)
                    sef_list.append(code_id)

                    spf_edges = self.__update_edges(spf_list, spf_edges, tr_id)
                    sef_edges = self.__update_edges(sef_list, sef_edges, tr_id)
        return spf_edges, sef_edges

    @classmethod
    def __update_edges(cls, _list, edges, tr_id):
        for depth, start_point in enumerate(_list):
            if depth == len(_list) - 1:
                break
            if depth >= len(edges):
                edges.append({})
            if start_point not in edges[depth]:
                edges[depth][start_point] = {}
            end_point = _list[depth + 1]
            if end_point not in edges[depth][start_point]:
                edges[depth][start_point][end_point] = set()
            edges[depth][start_point][end_point].add(tr_id)
        return edges

    def __rasterize(self, coordinate):
        x, y = coordinate[0], coordinate[1]
        xx = math.floor((x - self.origin_point[0]) / self.side_length)
        yy = math.floor((y - self.origin_point[1]) / self.side_length)
        return xx, yy


class IndexTree:
    """构建索引树"""

    def __init__(self, tr_dict):
        self.tr_dict = tr_dict
        self.spf_rt = None
        self.sef_rt = None

    def build(self, origin_point, side_length):
        edge_extract = EdgeExtract(self.tr_dict, origin_point, side_length)
        spf_edges, sef_edges = edge_extract.extract()
        self.spf_rt = self.__build_tree(spf_edges)
        self.sef_rt = self.__build_tree(sef_edges)

    @classmethod
    def __build_tree(cls, edges):
        rt = IndexTreeNode(name='RT', depth=0, tr_id_set=set([]), children={})
        for end_point_name, tr_id_set in edges[0][rt.name].items():
            rt.tr_id_set |= tr_id_set
            child = IndexTreeNode(parent=rt, children={}, name=end_point_name,
                                  depth=rt.depth+1, tr_id_set=tr_id_set)
            rt.children[end_point_name] = child
        rt.sort_children()

        node_list = [rt]
        while len(node_list) != 0:
            cur_node = node_list[0]
            node_list = node_list[1:]

            depth = cur_node.depth
            start_point_name = cur_node.name

            if depth >= len(edges):
                continue
            end_point_dict = edges[depth][start_point_name]

            for end_point_name, tr_id_set in end_point_dict.items():
                child = IndexTreeNode(parent=cur_node, name=end_point_name,
                                      children={}, depth=depth + 1, tr_id_set=tr_id_set)
                cur_node.children[end_point_name] = child
                node_list.append(child)
            cur_node.sort_children()

        return rt


def tree_output(node):
    print('      ' * node.depth, end='')
    # print(node.name, node.tr_id_set)
    print(node.name)
    for node_name, child in node.children.items():
        tree_output(child)

