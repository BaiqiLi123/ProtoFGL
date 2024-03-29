import collections
import random
import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


import numpy as np
from torch_geometric.transforms import BaseTransform
from Preprocess.base_splitter import BaseSplitter
import torch
from torch_geometric.utils import to_networkx, from_networkx
from utils import *
import networkx as nx

class COPRA:
    def __init__(self, G, T, v):
        """
        :param G:图本身
        :param T: 迭代次数T
        :param r:满足社区次数要求的阈值r
        """
        self._G = G
        self._n = len(G.nodes(False))  # 节点数目
        self._T = T
        self._v = v

    def execute(self):
        # 建立成员标签记录
        # 节点将被分配隶属度大于阈值的社区标签
        lablelist = {i: {i: 1} for i in self._G.nodes()}
        for t in range(self._T):
            visitlist = list(self._G.nodes())
            # 随机排列遍历顺序
            np.random.shuffle(visitlist)
            # 开始遍历节点
            for visit in visitlist:
                temp_count = 0
                temp_label = {}
                total = len(self._G[visit])
                # 根据邻居利用公式计算标签
                for i in self._G.neighbors(visit):
                    res = {key: value / total for key, value in lablelist[i].items()}
                    temp_label = dict(Counter(res) + Counter(temp_label))
                temp_count = len(temp_label)
                temp_label2 = temp_label.copy()
                for key, value in list(temp_label.items()):
                    if value < 1 / self._v:
                        del temp_label[key]
                        temp_count -= 1
                # 如果一个节点中所有的标签都低于阈值就随机选择一个
                if temp_count == 0:
                    # temp_label = {}
                    # v = self._v
                    # if self._v > len(temp_label2):
                    #     v = len(temp_label2)
                    # b = random.sample(temp_label2.keys(), v)
                    # tsum = 0.0
                    # for i in b:
                    #     tsum += temp_label2[i]
                    # temp_label = {i: temp_label2[i]/tsum for i in b}
                    if len(temp_label2.keys())!=0:
                        b = random.sample(temp_label2.keys(), 1)
                        temp_label = {b[0]: 1}
                    else:
                        print("err sampling")
                # 否则标签个数一定小于等于v个 进行归一化即可
                else:
                    tsum = sum(temp_label.values())
                    temp_label = {key: value / tsum for key, value in temp_label.items()}
                lablelist[visit] = temp_label

        communities = collections.defaultdict(lambda: list())
        # 扫描lablelist中的记录标签，相同标签的节点加入同一个社区中
        for primary, change in lablelist.items():
            for label in change.keys():
                communities[label].append(primary)
        # 返回值是个数据字典，value以集合的形式存在
        return communities.values()


def cal_EQ(cover, G):
    m = len(G.edges(None, False))  # 如果为真，则返回3元组（u、v、ddict）中的边缘属性dict。如果为false，则返回2元组（u，v）
    # 存储每个节点所在的社区
    vertex_community = collections.defaultdict(lambda: set())
    # i为社区编号(第几个社区) c为该社区中拥有的节点
    for i, c in enumerate(cover):
        # v为社区中的某一个节点
        for v in c:
            # 根据节点v统计他所在的社区i有哪些
            vertex_community[v].add(i)
    total = 0.0
    for c in cover:
        for i in c:
            # o_i表示i节点所同时属于的社区数目
            o_i = len(vertex_community[i])
            # k_i表示i节点的度数(所关联的边数)
            k_i = len(G[i])
            for j in c:
                t = 0.0
                # o_j表示j节点所同时属于的社区数目
                o_j = len(vertex_community[j])
                # k_j表示j节点的度数(所关联的边数)
                k_j = len(G[j])
                if G.has_edge(i, j):
                    t += 1.0 / (o_i * o_j)
                t -= k_i * k_j / (2 * m * o_i * o_j)
                total += t
    return round(total / (2 * m), 4)


def cal_Q(partition, G):  # 计算Q
    m = len(G.edges(None, False))  # 如果为真，则返回3元组（u、v、ddict）中的边缘属性dict。如果为false，则返回2元组（u，v）
    # print(G.edges(None,False))
    # print("=======6666666")
    a = []
    e = []
    for community in partition:  # 把每一个联通子图拿出来
        t = 0.0
        for node in community:  # 找出联通子图的每一个顶点
            t += len([x for x in G.neighbors(node)])  # G.neighbors(node)找node节点的邻接节点
        a.append(t / (2 * m))
    #             self.zidian[t/(2*m)]=community
    for community in partition:
        t = 0.0
        for i in range(len(community)):
            for j in range(len(community)):
                if (G.has_edge(community[i], community[j])):
                    t += 1.0
        e.append(t / (2 * m))

    q = 0.0
    for ei, ai in zip(e, a):
        q += (ei - ai ** 2)
    return q


def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G


# if __name__ == '__main__':
#     # G = nx.karate_club_graph()
#     G = load_graph('data/dolphin.txt')
#     start_time = time.time()
#     algorithm = COPRA(G, 20, 3)

#     communities = algorithm.execute()
#     end_time = time.time()
#     for i, community in enumerate(communities):
#         print(i, community)

#     print(cal_EQ(communities, G))
#     print(f'算法执行时间{end_time - start_time}')



class COPRASplitter(BaseSplitter):

    def __init__(self, client_num,  T, v , delta=20):
        self.client_num = client_num
        self.T = T
        self.v = v
        self.delta = delta
        super(COPRASplitter, self).__init__(client_num)


    def __call__(self, data, prior=None, **kwargs):
        label = data.y.numpy()
        data.index_orig = torch.arange(data.num_nodes)
        
        G = to_networkx(
            data,
            node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
            to_undirected=True)
        
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")
        
        algorithm = COPRA(G, self.T, self.v)
        
        idx_slice = list(algorithm.execute())

        # temp_idx_slice = list(algorithm.execute())
        # print(len(temp_idx_slice))
        # #在同一社区内继续划分
        # idx_slice = []
        # for i in range(len(temp_idx_slice)):
        #     slice_idx = len(temp_idx_slice[i])//4
        #     if slice_idx > 1:
        #         idx_slice.append(temp_idx_slice[i][:slice_idx])
        #         idx_slice.append(temp_idx_slice[i][slice_idx:2*slice_idx])
        #         idx_slice.append(temp_idx_slice[i][2*slice_idx:3*slice_idx])
        #         idx_slice.append(temp_idx_slice[i][3*slice_idx:])
        #     else:
        #         idx_slice.append(temp_idx_slice[i][:])

        nodes_sum = 0
        cluster2node = {}

        for i in range(len(idx_slice)):
            cluster2node[i] = idx_slice[i]
            nodes_sum += len(idx_slice[i])
        

        max_len = nodes_sum // self.client_num - self.delta
        max_len_client = nodes_sum // self.client_num


        tmp_cluster2node = {}
        for cluster in cluster2node:
            while len(cluster2node[cluster]) > max_len:
                tmp_cluster = cluster2node[cluster][:max_len]
                tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) + 1] = tmp_cluster
                cluster2node[cluster] = cluster2node[cluster][max_len:]
        cluster2node.update(tmp_cluster2node)

        orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
        orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

        client_node_idx = {idx: [] for idx in range(self.client_num)}
        idx = 0
        for (cluster, node_list) in orderedc2n:
            while len(node_list) + len(client_node_idx[idx]) > max_len_client + self.delta:
                idx = (idx + 1) % self.client_num
            client_node_idx[idx] += node_list
            idx = (idx + 1) % self.client_num  
        
        graphs = []


        client_all_node = getallClassnode(label, self.client_num)
        for owner in client_node_idx:
            client_node_idx[owner].extend(client_all_node[owner])


        for owner in client_node_idx:
            nodes = list(np.unique(client_node_idx[owner]))
            # nodes = list(set(client_node_idx[owner]))
            graphs.append(from_networkx(nx.subgraph(G, nodes)))

        return graphs

        