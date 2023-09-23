import numpy as np
import networkx as nx

from torch_geometric.transforms import BaseTransform
from Preprocess.base_splitter import BaseSplitter
import torch
from torch_geometric.utils import to_networkx, from_networkx
from collections import defaultdict
import networkx as nx

def sigm(x):
    # sigmoid操作 求梯度会用到
    # numpy.divide数组对应位置元素做除法。
    return np.divide(np.exp(-1. * x), 1. - np.exp(-1. * x))

def log_likelihood(F, A):
    """implements equation 2 of 
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf"""
    A_soft = F.dot(F.T)

    # Next two lines are multiplied with the adjacency matrix, A
    # A is a {0,1} matrix, so we zero out all elements not contributing to the sum
    FIRST_PART = A*np.log(1.-np.exp(-1.*A_soft))
    sum_edges = np.sum(FIRST_PART)
    SECOND_PART = (1-A)*A_soft
    sum_nedges = np.sum(SECOND_PART)

    log_likeli = sum_edges - sum_nedges
    return log_likeli
# def log_likelihood(F, A):
#     # 代入计算公式计算log似然度
#     A_soft = F.dot(F.T)

#     # 用邻接矩阵可以帮助我们只取到相邻的两个节点
#     FIRST_PART = A * np.log(1. - np.exp(-1. * A_soft))
#     sum_edges = np.sum(FIRST_PART)
#     # 1-A取的不相邻的节点
#     SECOND_PART = (1 - A) * A_soft
#     sum_nedges = np.sum(SECOND_PART)

#     log_likeli = sum_edges - sum_nedges
#     return log_likeli

def gradient(F, A, i):
    """Implements equation 3 of
    https://cs.stanford.edu/people/jure/pubs/bigclam-wsdm13.pdf
    
      * i indicates the row under consideration
    
    The many forloops in this function can be optimized, but for
    educational purposes we write them out clearly
    """
    N, C = F.shape

    neighbours = np.where(A[i])
    nneighbours = np.where(1-A[i])

    sum_neigh = np.zeros((C,))
    for nb in neighbours[0]:
        dotproduct = F[nb].dot(F[i])
        sum_neigh += F[nb]*sigm(dotproduct)

    sum_nneigh = np.zeros((C,))
    #Speed up this computation using eq.4
    for nnb in nneighbours[0]:
        sum_nneigh += F[nnb]

    grad = sum_neigh - sum_nneigh
    return grad
# def gradient(F, A, i):
#     # 代入公式计算梯度值
#     N, C = F.shape

#     # 通过邻接矩阵找到相邻 和 不相邻节点
#     neighbours = np.where(A[i])
#     nneighbours = np.where(1 - A[i])

#     # 公式第一部分
#     sum_neigh = np.zeros((C,))
#     for nb in neighbours[0]:
#         dotproduct = F[nb].dot(F[i])
#         sum_neigh += F[nb] * sigm(dotproduct)

#     # 公式第二部分
#     sum_nneigh = np.zeros((C,))
#     # Speed up this computation using eq.4
#     for nnb in nneighbours[0]:
#         sum_nneigh += F[nnb]

#     grad = sum_neigh - sum_nneigh
#     return grad

def train(A, C, iterations = 20):
    # initialize an F
    N = A.shape[0]
    F = np.random.rand(N,C)

    for n in range(iterations):
        for person in range(N):
            grad = gradient(F, A, person)

            F[person] += 0.005*grad

            F[person] = np.maximum(0.001, F[person]) # F should be nonnegative
        ll = log_likelihood(F, A)
        print('At step %5i/%5i ll is %5.3f'%(n, iterations, ll))
    return F
# def train(A, C, iterations=100):
#     # 初始化F
#     N = A.shape[0]
#     F = np.random.rand(N, C)

#     # 梯度下降最优化F
#     for n in range(iterations):
#         for person in range(N):
#             grad = gradient(F, A, person)

#             F[person] += 0.005 * grad

#             F[person] = np.maximum(0.001, F[person])  # F应该大于0
#         ll = log_likelihood(F, A)
#         print('At step %5i/%5i ll is %5.3f' % (n, iterations, ll))
#     return F


# 加载图数据集
def load_graph(path):
    G = nx.Graph()
    with open(path, 'r') as text:
        for line in text:
            vertices = line.strip().split(' ')
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G


class BIGCLAMSplitter(BaseSplitter):

    def __init__(self, client_num, overlapping_rate = 0, delta = 20, thre = 0.001):
        self.client_num = client_num
        self.thre = thre
        self.delta = delta
        self.ovlap = overlapping_rate
        super(BIGCLAMSplitter, self).__init__(client_num)


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
        

        adj = nx.to_numpy_array(G)

        F = train(adj, self.client_num)

        for line in F:
            print(line)

        idx_slice = defaultdict(list)
        for node in range(len(F)):
            for cla in range(len(F[node])):
                if F[node][cla] > self.thre:
                    idx_slice[cla].append(node)

        
        # # idx_slice = list(algorithm.execute())

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

        # # print(len(idx_slice))

        # nodes_sum = 0
        # cluster2node = {}

        # for i in range(len(idx_slice)):
        #     cluster2node[i] = idx_slice[i]
        #     nodes_sum += len(idx_slice[i])
        

        # max_len = nodes_sum // self.client_num - self.delta
        # max_len_client = nodes_sum // self.client_num


        # tmp_cluster2node = {}
        # for cluster in cluster2node:
        #     while len(cluster2node[cluster]) > max_len:
        #         tmp_cluster = cluster2node[cluster][:max_len]
        #         tmp_cluster2node[len(cluster2node) + len(tmp_cluster2node) + 1] = tmp_cluster
        #         cluster2node[cluster] = cluster2node[cluster][max_len:]
        # cluster2node.update(tmp_cluster2node)

        # orderedc2n = (zip(cluster2node.keys(), cluster2node.values()))
        # orderedc2n = sorted(orderedc2n, key=lambda x: len(x[1]), reverse=True)

        # client_node_idx = {idx: [] for idx in range(self.client_num)}
        # idx = 0
        # for (cluster, node_list) in orderedc2n:
        #     while len(node_list) + len(client_node_idx[idx]) > max_len_client + self.delta:
        #         idx = (idx + 1) % self.client_num
        #     client_node_idx[idx] += node_list
        #     idx = (idx + 1) % self.client_num  
            
        # graphs = []
        # for owner in client_node_idx:
        #     nodes = list(set(client_node_idx[owner]))
        #     graphs.append(from_networkx(nx.subgraph(G, nodes)))

        graphs = []
        for owner in idx_slice:
            # nodes = list(set(idx_slice[owner]))
            nodes = list(np.unique(idx_slice[owner]))
            graphs.append(from_networkx(nx.subgraph(G, nodes)))

        return graphs