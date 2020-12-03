import numpy as np
import pandas as pd


class IndexTreeNode:

    def __init__(
        self,
        name,
        parent,
        depth,
        children={},
        tids=set(),
    ):
        self.name = name
        self.parent = parent
        self.children = children
        self.tids = tids
        self.depth = depth


class IndexTree:

    def __init__(self, f_df):
        self.f_df = f_df
        self.rt = None

    def build(self):
        self.rt = IndexTreeNode(
            name='RT',
            parent=None,
            depth=0,
            children={},
            tids=set()
        )
        self.__extend_children(self.rt, self.f_df)

    def __extend_children(self, cur_node, cur_level_df):

        cur_node.tids = set(np.unique(cur_level_df.values))

        if cur_node.depth >= 4:
            return
        for name in cur_level_df.index.get_level_values(0).unique():

            child = IndexTreeNode(
                name=name,
                parent=cur_node,
                depth=cur_node.depth + 1,
                children={},
                tids=set()
            )

            cur_node.children[name] = child
            self.__extend_children(child, cur_level_df.loc[name])

    def query(self, query_tree):

        tids = set()

        base_nodes, query_nodes = [self.rt], [query_tree.rt]
        while base_nodes != []:
            b_node, q_node = base_nodes[0], query_nodes[0]
            base_nodes, query_nodes = base_nodes[1:], query_nodes[1:]

            flag = True
            for name, child in q_node.children.items():
                if name in b_node.children:
                    base_nodes.append(b_node.children[name])
                    query_nodes.append(child)
                    flag = False
            if flag:
                tids |= b_node.tids
        return tids

    def update(self, new_tree):

        base_nodes, new_nodes = [self.rt], [new_tree.rt]
        while base_nodes != []:
            b_node, n_node = base_nodes[0], new_nodes[0]
            base_nodes, new_nodes = base_nodes[1:], new_nodes[1:]

            b_node.tids |= n_node.tids

            for name, child in n_node.children.items():
                if name in b_node.children:
                    base_nodes.append(b_node.children[name])
                    new_nodes.append(child)
                else:
                    b_node.children[name] = child

    def print_tree(self):
        self.dfs_tree(self.rt)

    def dfs_tree(self, node):
        print('-' * 10 * node.depth, node.name)
        for _, child in node.children.items():
            self.dfs_tree(child)
