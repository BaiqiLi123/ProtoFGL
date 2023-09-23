import numpy as np
from Preprocess.base_splitter import BaseSplitter
import torch
from torch_geometric.utils import to_networkx, from_networkx

import networkx as nx

from utils import getallClassnode


def _split_according_to_prior(label, client_num, prior):
    assert client_num == len(prior)
    classes = len(np.unique(label))
    assert classes == len(np.unique(np.concatenate(prior, 0)))

    # counting
    frequency = np.zeros(shape=(client_num, classes))
    for idx, client_prior in enumerate(prior):
        for each in client_prior:
            frequency[idx][each] += 1
    sum_frequency = np.sum(frequency, axis=0)

    idx_slice = [[] for _ in range(client_num)]
    for k in range(classes):
        idx_k = np.where(label == k)[0]
        np.random.shuffle(idx_k)
        nums_k = np.ceil(frequency[:, k] / sum_frequency[k] *
                         len(idx_k)).astype(int)
        while len(idx_k) < np.sum(nums_k):
            random_client = np.random.choice(range(client_num))
            if nums_k[random_client] > 0:
                nums_k[random_client] -= 1
        assert len(idx_k) == np.sum(nums_k)
        idx_slice = [
            idx_j + idx.tolist() for idx_j, idx in zip(
                idx_slice, np.split(idx_k,
                                    np.cumsum(nums_k)[:-1]))
        ]

    for i in range(len(idx_slice)):
        np.random.shuffle(idx_slice[i])
    return idx_slice


def dirichlet_distribution_noniid_slice(label,
                                        client_num,
                                        alpha,
                                        min_size=1,
                                        prior=None):
    r"""Get sample index list for each client from the Dirichlet distribution.
    https://github.com/FedML-AI/FedML/blob/master/fedml_core/non_iid
    partition/noniid_partition.py

    Arguments:
        label (np.array): Label list to be split.
        client_num (int): Split label into client_num parts.
        alpha (float): alpha of LDA.
        min_size (int): min number of sample in each client
    Returns:
        idx_slice (List): List of splited label index slice.
    """
    if len(label.shape) != 1:
        raise ValueError('Only support single-label tasks!')

    if prior is not None:
        return _split_according_to_prior(label, client_num, prior)

    num = len(label)
    classes = len(np.unique(label))
    assert num > client_num * min_size, f'The number of sample should be ' \
                                        f'greater than' \
                                        f' {client_num * min_size}.'
    size = 0
    while size < min_size:
        idx_slice = [[] for _ in range(client_num)]
        for k in range(classes):
            # for label k
            idx_k = np.where(label == k)[0]
            np.random.shuffle(idx_k)
            prop = np.random.dirichlet(np.repeat(alpha, client_num))
            # prop = np.array([
            #    p * (len(idx_j) < num / client_num)
            #    for p, idx_j in zip(prop, idx_slice)
            # ])
            # prop = prop / sum(prop)
            prop = (np.cumsum(prop) * len(idx_k)).astype(int)[:-1]
            idx_slice = [
                idx_j + idx.tolist()
                for idx_j, idx in zip(idx_slice, np.split(idx_k, prop))
            ]
            size = min([len(idx_j) for idx_j in idx_slice])
    for i in range(client_num):
        np.random.shuffle(idx_slice[i])
    return idx_slice




class LDASplitter(BaseSplitter):
    """
    This splitter split dataset with LDA.

    Args:
        client_num: the dataset will be split into ``client_num`` pieces
        alpha (float): Partition hyperparameter in LDA, smaller alpha \
            generates more extreme heterogeneous scenario see \
            ``np.random.dirichlet``
    """
    def __init__(self, client_num, alpha=None):
        self.alpha = alpha
        super(LDASplitter, self).__init__(client_num)

#     def __call__(self, dataset, prior=None, **kwargs):
#         from torch.utils.data import Dataset, Subset

#         tmp_dataset = [ds for ds in dataset]
#         label = np.array([y for x, y in tmp_dataset])
#         idx_slice = dirichlet_distribution_noniid_slice(label,
#                                                         self.client_num,
#                                                         self.alpha,
#                                                         prior=prior)
#         if isinstance(dataset, Dataset):
#             data_list = [Subset(dataset, idxs) for idxs in idx_slice]
#         else:
#             data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice]
#         return data_list
    
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
        
        idx_slice = dirichlet_distribution_noniid_slice(label,
                                                        self.client_num,
                                                        self.alpha,
                                                        prior=prior)
        
        graphs = []

        # client_all_node = getallClassnode(label, self.client_num)
        # for owneri in range(len(idx_slice)):
        #     idx_slice[owneri].extend(client_all_node[owneri])

        for index in range(self.client_num):
            nodes = idx_slice[index]
            graphs.append(from_networkx(nx.subgraph(G, nodes)))

        return graphs
        
        
        