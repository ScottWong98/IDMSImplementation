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

    def output(self):
        self.dfs(self.rt)

    def dfs(self, node):
        print('      ' * node.depth, end='')
        # print(node.name, node.tr_id_set)
        print(node.name)
        for node_name, child in node.children.items():
            self.dfs(child)

