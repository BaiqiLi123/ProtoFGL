import scipy.sparse as sp
from base_model import *

from torch_geometric.data import Data
import torch
from torch_geometric.utils import add_self_loops, remove_self_loops, to_undirected, to_dense_adj
from Preprocess.mycora import *

from collections import defaultdict


def getClassdict(labels_list, train_idx, val_idx, test_idx):
    class_train_dict = defaultdict(list)
    class_test_dict = defaultdict(list)
    class_val_dict = defaultdict(list)
    for idx in train_idx:
        class_train_dict[labels_list[idx]].append(idx)
    for idx in val_idx:
        class_val_dict[labels_list[idx]].append(idx)    
    for idx in test_idx:
        class_test_dict[labels_list[idx]].append(idx)    
    
    class_train_dict = {key: class_train_dict[key] for key in sorted(class_train_dict.keys())}
    class_val_dict = {key: class_val_dict[key] for key in sorted(class_val_dict.keys())}
    class_test_dict = {key: class_test_dict[key] for key in sorted(class_test_dict.keys())}
    
    
    return class_train_dict, class_test_dict, class_val_dict


def transformData(data_local_dict, train_prpoportion=0.6, val_prpoportion=0.2, test_prpoportion=0.2):
        
    for idx in data_local_dict:
        data = data_local_dict[idx]
        
        data.edge_index = add_self_loops(
            to_undirected(remove_self_loops(data.edge_index)[0]),
            num_nodes=data.x.shape[0])[0]

        data.x = torch.tensor(normalize(data.x))

        adj = to_dense_adj(data.edge_index).squeeze()

        local_idx = np.arange(data.num_nodes)
        shuffle_idx = np.random.shuffle(local_idx)
        
        train_idx = local_idx[:int(len(local_idx)*train_prpoportion)]
        val_idx = local_idx[int(len(local_idx)*train_prpoportion): int(len(local_idx)*(train_prpoportion + val_prpoportion))]
        test_idx = local_idx[int(len(local_idx)*(train_prpoportion + val_prpoportion)):]

        data.train_mask = torch.zeros_like(data.train_mask)
        data.val_mask = torch.zeros_like(data.val_mask)
        data.test_mask = torch.zeros_like(data.test_mask)

        data.train_mask[train_idx] = True
        data.val_mask[val_idx] = True
        data.test_mask[test_idx] = True

        
        class_train_dict, class_test_dict, class_val_dict = getClassdict(data.y.numpy(), train_idx, val_idx, test_idx)
        data.adj = adj
        data.train_idx = train_idx
        data.test_idx = test_idx
        data.val_idx = val_idx
        data.class_train_dict = class_train_dict
        data.class_test_dict = class_test_dict
        data.class_val_dict = class_val_dict      
    
    global_data = data_local_dict.pop(0)
    clients_data = {}
    for i in data_local_dict:
        clients_data[i-1] = data_local_dict[i]

    # Keep ML split consistent with local graphs
    if global_data is not None:
        train_mask = torch.zeros_like(global_data.train_mask)
        val_mask = torch.zeros_like(global_data.val_mask)
        test_mask = torch.zeros_like(global_data.test_mask)

        for client_sampler in data_local_dict.values():
            if isinstance(client_sampler, Data):
                client_subgraph = client_sampler
            else:
                client_subgraph = client_sampler['data']
            train_mask[client_subgraph.index_orig[client_subgraph.train_mask]] = True
            val_mask[client_subgraph.index_orig[client_subgraph.val_mask]] = True
            test_mask[client_subgraph.index_orig[client_subgraph.test_mask]] = True

        nodes_index = np.arange(global_data.x.shape[0])
        global_data.train_idx = list(copy.deepcopy(nodes_index[train_mask]))
        global_data.val_idx = list(copy.deepcopy(nodes_index[val_mask]))
        global_data.test_idx = list(copy.deepcopy(nodes_index[test_mask]))

        global_data.class_train_dict, global_data.class_test_dict, global_data.class_val_dict = getClassdict(global_data.y.numpy(), global_data.train_idx, global_data.val_idx, global_data.test_idx)

        global_data.train_mask = train_mask
        global_data.val_mask = val_mask
        global_data.test_mask = test_mask

    return global_data, clients_data

def load_data(dataset, splitter, root, client_num,alpha=None):
    if dataset == 'cora' or True:#no use
        if splitter == 'random':
            data_local_dict, _ = my_cora_RandomSplit(root, client_num, config=None,datasetname=dataset)
        elif splitter == 'Louvain':
            data_local_dict, _ = my_cora_LouvainSplit(root, client_num, config=None,datasetname=dataset)
        elif splitter == 'LDA':
            data_local_dict, _ = my_cora_LDASplit(root, client_num, alpha=alpha,config=None,datasetname=dataset)
        elif splitter == 'corpa':
            data_local_dict, _ = my_cora_CORPASplit(root, client_num, k = 20, v= 3, config=None,datasetname=dataset)
        elif splitter == 'bigclam':
            data_local_dict, _  = my_cora_BIGCLAMSplit(root, client_num,datasetname=dataset)
    global_data, clients_data =  transformData(data_local_dict)  
    return global_data, clients_data

def normalize_adj(adj_m):
    """Row-normalize  matrix"""
    mx = adj_m.copy()
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = np.diag(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def normalize_features(feat):
    """Row-normalize sparse matrix"""
    mx = feat.copy()
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)