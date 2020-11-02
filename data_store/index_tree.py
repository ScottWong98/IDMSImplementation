import time
from data_processing.data_process import DataProcess


class IndexTreeNode:

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


class IndexTree:

    def __init__(self, raw_edges):
        self.raw_edges = raw_edges
        self.rt = None

    def build(self):
        self.rt = IndexTreeNode(name='RT', depth=0, tr_id_set=set([]),children={})
        raw_edges = self.raw_edges
        for ep_name, tr_id_set in raw_edges[0][self.rt.name].items():
            self.rt.tr_id_set |= tr_id_set
            child = IndexTreeNode(parent=self.rt,
                                  children={},
                                  name=ep_name,
                                  depth=self.rt.depth + 1,
                                  tr_id_set=tr_id_set)
            self.rt.children[ep_name] = child
        self.rt.sort_children()

        node_list = [self.rt]
        while len(node_list) != 0:
            cur_node = node_list[0]
            node_list = node_list[1:]

            depth = cur_node.depth
            sp_name = cur_node.name

            if depth >= len(raw_edges):
                continue
            ep_dict = raw_edges[depth][sp_name]

            # print(ep_dict)

            for ep_name, tr_id_set in ep_dict.items():
                child = IndexTreeNode(parent=cur_node, name=ep_name, children={}, depth=depth+1, tr_id_set=tr_id_set)
                cur_node.children[ep_name] = child
                node_list.append(child)
            cur_node.sort_children()

    def query(self, qt_rt):
        result = set()

        node_list = [self.rt]
        qt_node_list = [qt_rt]
        while len(node_list) != 0:
            node = node_list[0]
            # print(len(node_list))
            qt_node = qt_node_list[0]

            node_list = node_list[1:]
            qt_node_list = qt_node_list[1:]

            flag = True
            for (name, child) in qt_node.children.items():
                if name in node.children:
                    node_list.append(node.children[name])
                    qt_node_list.append(child)
                    flag = False

            if flag:
                result |= node.tr_id_set

        return result

    def output(self):
        self.dfs(self.rt)

    def dfs(self, node):
        print('      ' * node.depth, end='')
        # print(node.name, node.tr_id_set)
        print(node.name)
        for node_name, child in node.children.items():
            self.dfs(child)


if __name__ == '__main__':
    start = time.process_time()
    data_process = DataProcess("../resource/test_12_user_with_flag.csv", 0.01, 5000)
    data_process.run()
    space_first_edges, semantic_first_edges = data_process.extract_edges(origin_point=(190, 190), side_length=5)
    space_first_tree = IndexTree(raw_edges=space_first_edges)
    space_first_tree.build()
    semantic_first_tree = IndexTree(raw_edges=semantic_first_edges)
    semantic_first_tree.build()

    query_data_process = DataProcess("../resource/test_1_user_with_flag.csv", 0.01, 5000)
    query_data_process.run()
    query_space_fist_edges, query_semantic_first_edges = data_process.extract_edges(origin_point=(190, 190), side_length=5)
    query_space_first_tree = IndexTree(raw_edges=query_space_fist_edges)
    query_space_first_tree.build()
    # query_space_first_tree.output()

    similar_tr = space_first_tree.query(query_space_first_tree.rt)
    print(similar_tr)
    end = time.process_time()
    print("Finish all in %s " % str(end - start))
