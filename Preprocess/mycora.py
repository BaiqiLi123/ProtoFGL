import numpy as np

from torch_geometric.datasets import Planetoid

from Preprocess.louvain_splitter import LouvainSplitter
from Preprocess.random_splitter import RandomSplitter

from Preprocess.splitter import *
from Preprocess.overlapp_splitter import *
from Preprocess.BigClam import *
'''
from analyzer import Analyzer
'''
# from federatedscope.core.splitters.graph import LouvainSplitter
# from federatedscope.register import register_data

def my_cora_LDASplit(root, client_num, config=None,datasetname="",alpha=None):
    if config:
        path = config.data.root
        client_num = config.federate.client_num
    else:
        path = root
        client_num = client_num
        

    num_split = [232, 542, np.iinfo(np.int64).max]
    dataset = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])
    global_data = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])[0]
    global_data.index_orig = np.arange(global_data.num_nodes)
    dataset = LDASplitter(client_num,alpha=alpha)(dataset[0])

    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        data_local_dict[client_idx + 1] = dataset[client_idx]

    data_local_dict[0] = global_data
    return data_local_dict, config

def my_cora_LouvainSplit(root, client_num, config=None,datasetname=""):
    if config:
        path = config.data.root
        client_num = config.federate.client_num
    else:
        path = root
        client_num = client_num
        

    num_split = [232, 542, np.iinfo(np.int64).max]
    dataset = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])
    global_data = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])[0]
    global_data.index_orig = np.arange(global_data.num_nodes)
    dataset = LouvainSplitter(client_num)(dataset[0])

    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        data_local_dict[client_idx + 1] = dataset[client_idx]

    data_local_dict[0] = global_data
    return data_local_dict, config

def my_cora_RandomSplit(root, client_num, config=None,datasetname=""):
    if config:
        path = config.data.root
        client_num = config.federate.client_num
    else:
        path = root
        client_num = client_num

    num_split = [232, 542, np.iinfo(np.int64).max]
    dataset = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])
    global_data = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])[0]
    global_data.index_orig = np.arange(global_data.num_nodes)
    dataset = RandomSplitter(client_num)(dataset[0])

    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        data_local_dict[client_idx + 1] = dataset[client_idx]

    data_local_dict[0] = global_data
    return data_local_dict, config


def my_cora_CORPASplit(root, client_num, k = 20, v =3, config=None,datasetname=""):
    if config:
        path = config.data.root
        client_num = config.federate.client_num
    else:
        path = root
        client_num = client_num
        

    num_split = [250, 650, np.iinfo(np.int64).max]
    dataset = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])
    global_data = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])[0]
    global_data.index_orig = np.arange(global_data.num_nodes)
    dataset = COPRASplitter(client_num, k, v)(dataset[0])
    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        data_local_dict[client_idx + 1] = dataset[client_idx]

    data_local_dict[0] = global_data
    return data_local_dict, config


def my_cora_BIGCLAMSplit(root, client_num, config=None,datasetname=""):
    if config:
        path = config.data.root
        client_num = config.federate.client_num
    else:
        path = root
        client_num = client_num
        

    num_split = [200, 650, np.iinfo(np.int64).max]
    dataset = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])
    global_data = Planetoid(path,
                        datasetname,
                        split='random',
                        num_train_per_class=num_split[0],
                        num_val=num_split[1],
                        num_test=num_split[2])[0]
    global_data.index_orig = dict([(nid, nid) for nid in range(global_data.num_nodes)])
    dataset = BIGCLAMSplitter(client_num)(dataset[0])
    data_local_dict = dict()
    for client_idx in range(len(dataset)):
        data_local_dict[client_idx + 1] = dataset[client_idx]

    data_local_dict[0] = global_data
    return data_local_dict, config
