import enum
import os
import time

import numpy as np
import torch
import json
from decimal import Decimal
import random
from sklearn.metrics import f1_score
import torch.nn.functional as F
import scipy.sparse as sp
import torch.nn as nn
import copy
from collections import defaultdict
def evalandprint(args, algclass, client_data, global_data, SAVE_PATH, best_acc, best_tacc, best_global_tacc, a_iter, best_changed, proto_modle = False):
    # evaluation on training data
    for client_idx in range(args.n_clients):
        train_loss, train_acc = algclass.client_eval(client_idx, client_data, mode = 'train')
        print(
            f' Site-{client_idx:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

    # evaluation on valid data
    val_acc_list = [None] * args.n_clients
    for client_idx in range(args.n_clients):
        val_loss, val_acc = algclass.client_eval(client_idx, client_data, mode = 'val')
        val_acc_list[client_idx] = val_acc
        print(
            f' Site-{client_idx:02d} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')

    if np.mean(val_acc_list) > np.mean(best_acc):
        for client_idx in range(args.n_clients):
            best_acc[client_idx] = val_acc_list[client_idx]
            best_epoch = a_iter
        best_changed = True

    if best_changed:
        best_changed = False
        # test
        for client_idx in range(args.n_clients):
            _, test_acc = algclass.client_eval(client_idx, client_data, mode = 'test')
            _, global_tacc = algclass.global_eval(client_idx, client_data, global_data)
            print(
                f' Test site-{client_idx:02d} | Epoch:{best_epoch} | Test Acc: {test_acc:.4f} | Global Test Acc: {global_tacc:.4f}')
            best_global_tacc[client_idx] = global_tacc
            best_tacc[client_idx] = test_acc
        print(f' Saving the local and server checkpoint to {SAVE_PATH}')
        tosave = {'best_epoch': best_epoch, 'best_acc': best_acc, 'best_tacc': np.mean(np.array(best_tacc)), 'best_global_tacc': np.mean(np.array(best_global_tacc))}
        if not proto_modle:
            for i,tmodel in enumerate(algclass.client_model):
                tosave['client_model_'+str(i)]=tmodel.state_dict()
            tosave['server_model']=algclass.server_model.state_dict()
        else:
            for i,tmodel in enumerate(algclass.client_model):
                for model in tmodel:
                    tosave['client_model_{}_{}'.format(i, model)]=tmodel[model].state_dict()
            for model in algclass.server_model:
                tosave['server_model_{}'.format(model)]=algclass.server_model[model].state_dict()            
        torch.save(tosave, SAVE_PATH)

    return best_acc, best_tacc, best_global_tacc,best_changed

def analyze_datasets(datasets):
    analyze_res={}
    all_nodes_sum = 0
    
    clients_num = len(datasets)
    
    nodes_num_for_clients = [0 for _ in range(clients_num)]
    
    overlap_node = {}
    
    overlap_node_sum_clients = [0 for _ in range(clients_num)] 
    
    class_proportion_for_clients = [{} for _ in range(clients_num)]
    train_class_proportion_for_clients = [{} for _ in range(clients_num)]
    val_class_proportion_for_clients = [{} for _ in range(clients_num)]
    test_class_proportion_for_clients = [{} for _ in range(clients_num)]

    class_cnt = [{} for _ in range(clients_num)] 
    class_edge_cnt = [{} for _ in range(clients_num)]

    
    for client_idx in range(clients_num):
        data = datasets[client_idx]

        nodes_num_for_clients[client_idx] = len(data.x)
        all_nodes_sum += len(data.x)
        
        index_orig = np.unique(data.index_orig.numpy())
        y = data.y.numpy()
        
        #print(data.edge_index)
        #print(data.adj)
        dim0, dim1 = data.adj.shape

        for i in range(dim0):
            for j in range(i+1,dim1):
                element = data.adj[i][j]
                if element>0:
                    #print(i,j)
                    try:
                        class_edge_cnt[client_idx][int(y[i])]+=1
                    except:
                        class_edge_cnt[client_idx][int(y[i])]=1
                    try:
                        class_edge_cnt[client_idx][int(y[i])] += 1
                    except:
                        class_edge_cnt[client_idx][int(y[i])] = 1


        
        for label in y:
            label=int(label)
            if label in class_proportion_for_clients[client_idx]:
                class_proportion_for_clients[client_idx][label] += 1
            else:
                class_proportion_for_clients[client_idx][label] = 1



        class_proportion_for_clients[client_idx] = dict(sorted(class_proportion_for_clients[client_idx].items(), key=lambda x: x[0]))  
        

        for label in y[data.train_mask]:
            if label in train_class_proportion_for_clients[client_idx]:
                train_class_proportion_for_clients[client_idx][label] += 1
            else:
                train_class_proportion_for_clients[client_idx][label] = 1

        for key in train_class_proportion_for_clients[client_idx]:
            train_class_proportion_for_clients[client_idx][key] = train_class_proportion_for_clients[client_idx][key]/(data.train_mask.sum().item())  

        train_class_proportion_for_clients[client_idx] = dict(sorted(train_class_proportion_for_clients[client_idx].items(), key=lambda x: x[0]))        
        

        for label in y[data.val_mask]:
            if label in val_class_proportion_for_clients[client_idx]:
                val_class_proportion_for_clients[client_idx][label] += 1
            else:
                val_class_proportion_for_clients[client_idx][label] = 1

        for key in val_class_proportion_for_clients[client_idx]:
            val_class_proportion_for_clients[client_idx][key] = val_class_proportion_for_clients[client_idx][key]/(data.val_mask.sum().item())        
        val_class_proportion_for_clients[client_idx] = dict(sorted(val_class_proportion_for_clients[client_idx].items(), key=lambda x: x[0]))        
        

        for label in y[data.test_mask]:
            if label in test_class_proportion_for_clients[client_idx]:
                test_class_proportion_for_clients[client_idx][label] += 1
            else:
                test_class_proportion_for_clients[client_idx][label] = 1

        for key in test_class_proportion_for_clients[client_idx]:
            test_class_proportion_for_clients[client_idx][key] = test_class_proportion_for_clients[client_idx][key]/(data.test_mask.sum().item())        
        test_class_proportion_for_clients[client_idx] = dict(sorted(test_class_proportion_for_clients[client_idx].items(), key=lambda x: x[0]))   
    class_cnt=class_proportion_for_clients
    analyze_res["class_cnt"]={i:class_cnt[i] for i in range(clients_num)}
    analyze_res["class_edge_cnt"] = {i:class_edge_cnt[i] for i in range(clients_num)}
    print(analyze_res)

    #filedir = 'LDA'+str(args.alpha)+' 50501010 5_200/' + args.dataset + "/"
    filedir="analy/"
    filename = str(time.time()) + '.json'
    os.makedirs(filedir, exist_ok=True)
    with open(filedir + filename, 'w') as f:
        json.dump(analyze_res, f, indent=4)

    unique_node_list = []
    for i in range(clients_num):
        unique_node_list.extend(datasets[i].index_orig)
    unique_node_list = np.unique(unique_node_list)
                       
    print("----------------------global-----------------")
    print('all nodes: {}'.format(all_nodes_sum))
    print('unique nodes: {}'.format(len(unique_node_list)))
    
    for idx in range(clients_num):
        for key in train_class_proportion_for_clients[idx]:
            print("train :  class: {}     propotion: {:.2f}".format(key ,train_class_proportion_for_clients[idx][key]))              
        
        for key in val_class_proportion_for_clients[idx]:
            print("val :  class: {}     propotion: {:.2f}".format(key ,val_class_proportion_for_clients[idx][key]))              
        
        for key in test_class_proportion_for_clients[idx]:
            print("test :  class: {}     propotion: {:.2f}".format(key ,test_class_proportion_for_clients[idx][key]))              
        

        for index in index_orig:
            if index not in unique_node_list:
                unique_node_list.append(index)
            else:
                if index in overlap_node:
                    overlap_node[index] += 1
                else:
                    overlap_node[index] = 2
        
                    
    for idx in range(clients_num):
        Sum = 0
        data = datasets[idx]
        index_orig = np.unique(data.index_orig.numpy())
        for i in index_orig:
            if i in overlap_node:
                Sum += 1
        overlap_node_sum_clients[idx] = Sum


def get_overlapping_for_client(datasets):  
    """

    Input datasets:data....

    Return matrix_count, matrix_proportion
    """   
    clients_num = len(datasets)

    
    matrix_count = [[0 for j in range(clients_num)] for i in range(clients_num)]
    matrix_proportion = [[0 for j in range(clients_num)] for i in range(clients_num)]

    sum_for_client = [0 for i in range(clients_num)]

    for x in range(clients_num):
        nodes_x = datasets[x].index_orig[datasets[x].train_mask]
        for y in range(clients_num):
            if y == x:
                continue
            else:
                nodes_y = datasets[y].index_orig[datasets[y].train_mask]

                for node in nodes_y:
                    if node in nodes_x:
                        matrix_count[x][y] += 1
                
            sum_for_client[x] += matrix_count[x][y]

        for y in range(clients_num):
            matrix_proportion[x][y] = round(matrix_count[x][y]/sum_for_client[x], 2)
    
    unique_node = []
    overlap_node = []
    count_overlap_node = [0 for i in range(clients_num)]

    for x in range(clients_num):
        nodes_x = datasets[x].index_orig[datasets[x].train_mask]
        for node in nodes_x:
            if node not in unique_node:
                unique_node.append(node)
            else:
                if node not in overlap_node:
                    overlap_node.append(node)
    
    for x in range(clients_num):
        nodes_x = datasets[x].index_orig[datasets[x].train_mask] 
        for node in nodes_x:
            if node in overlap_node:
                count_overlap_node[x] += 1
            

    print("matrix_count:")


    print("\n matrix_proportion:")
    for i in range(len(matrix_proportion)):
        print(matrix_proportion[i])

    return matrix_count, matrix_proportion

# Newly add
def class_list_generator(global_data, clients_data, args):
    global_class_list = {i: [] for i in range(args.n_way)}
    label = global_data.y.numpy()
    for i in range(global_data.x.shape[0]):
        global_class_list[label[i]].append(i)
    client_class_list = [{i: [] for i in range(args.n_way)} for _ in range(args.n_clients)]
    for i in range(args.n_clients):
        label = clients_data[i].y.numpy()
        for j in range(clients_data[i].x.shape[0]):
            client_class_list[i][label[j]].append(j)
    return global_class_list, client_class_list
    
def task_generator(id_by_class, class_list, n_way, k_shot, m_query):
    # sample class indices
    #class_selected = random.sample(class_list, len(class_list))
    class_selected = class_list
    id_support = []
    id_query = []
    for cla in class_selected:
        if k_shot + m_query <= len(id_by_class[cla]):
            temp = random.sample(id_by_class[cla], k_shot + m_query)
            id_support.extend(temp[:k_shot])
            id_query.extend(temp[k_shot:])
        else:
            # Newly added
            _k_shot = int(k_shot/(k_shot+m_query)*len(id_by_class[cla]))
            _m_query = int(m_query/(k_shot+m_query)*len(id_by_class[cla]))
            tmp_support = []
            tmp_query = []
            seq = np.array([i for i in range(len(id_by_class[cla]))])
            random.shuffle(seq)
            temp = list(np.array(id_by_class[cla])[seq])
            if _k_shot != 0:
                _id_support = temp[:_k_shot]
                for _ in range(int(k_shot/_k_shot)):
                    tmp_support.extend(_id_support)
                random.shuffle(tmp_support)
                tmp_support.extend(random.sample(_id_support, -len(tmp_support) + k_shot))
                id_support.extend(tmp_support)
            else:
                id_support.extend([id_by_class[cla][0] for _ in range(k_shot)])
            if _m_query != 0:
                _id_query = temp[_k_shot:]
                for _ in range(int(m_query / _m_query)):
                    tmp_query.extend(_id_query)
                random.shuffle(tmp_query)
                if -len(tmp_query) + m_query > 0:
                    tmp_query.extend(random.sample(_id_query, -len(tmp_query) + m_query))
                else:
                    tmp_query = tmp_query[:m_query]
                id_query.extend(tmp_query)
            else:
                id_query.extend([id_by_class[cla][-1] for _ in range(m_query)])

    return np.array(id_support), np.array(id_query), class_selected

# without class distinguishment
def _task_generator(length, k_shot, m_query):
    nodes = [i for i in range(length)]
    random.shuffle(nodes)
    id_support = []
    id_query = []
    if k_shot + m_query <= length:
        select = random.sample(nodes, k_shot + m_query)
        id_support = select[:k_shot]
        id_query = select[k_shot:]
    else:
        _k_shot = int(k_shot / (k_shot + m_query) * length)
        _m_query = int(m_query / (k_shot + m_query) * length)
        tmp_support = []
        tmp_query = []
        if _k_shot != 0:
            for _ in range(int(k_shot/_k_shot)):
                tmp_support.extend(nodes[:_k_shot])
            random.shuffle(tmp_support)
            tmp_support.extend(random.sample(nodes[:_k_shot], -len(tmp_support) + k_shot))
            id_support.extend(tmp_support)
        else:
            id_support.extend([0 for _ in range(k_shot)])
        # if _m_query != 0:
        #     for _ in range(int(m_query/_m_query)):
        #         tmp_query.extend(nodes[_k_shot:])
        #         random.shuffle(tmp_query)
        #     tmp_query.extend(random.sample(nodes[_k_shot:], -len(tmp_query) + m_query))
        #     id_query.extend(tmp_query)
        # else:
        #     id_query.extend([length-1 for _ in range(m_query)])

        if _m_query != 0:
            for _ in range(int(m_query / _m_query)):
                tmp_query.extend(nodes[_k_shot:])
                random.shuffle(tmp_query)
            if -len(tmp_query) + m_query >= 0:
                tmp_query.extend(random.sample(nodes[_k_shot:], -len(tmp_query) + m_query))
            id_query.extend(tmp_query)
        else:
            id_query.extend([length-1 for _ in range(m_query)])

    return np.array(id_support), np.array(id_query)

def getallClassnode(label, clientNum):
    class_node_dict = defaultdict(list)
    clients_node = {}
    for idx in range(len(label)):
        class_node_dict[label[idx]].append(idx)
    for client_idx in range(clientNum):
        clients_node[client_idx] = []
        for cla in class_node_dict.keys():
            clients_node[client_idx].extend(random.sample(class_node_dict[cla], 2))
    return clients_node

def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)  # N x M

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    #print('=============')
    #print(preds)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1

def train(model, optimizer, features, adj, labels, id_support, id_query):
    model.train()
    optimizer.zero_grad()
    out = model(features, adj)
    loss_train = F.nll_loss(out[id_query], labels[id_query])
    loss_train.backward()
    optimizer.step()

def train_prox(model, optimizer, server_model, features, adj, labels, id_support, id_query):
    model.train()
    # global_model = copy.deepcopy(server_model)
    optimizer.zero_grad()
    out = model(features, adj)
    loss_train = F.nll_loss(out[id_query], labels[id_query])
    proximal_term = 0.0
    for w, w_t in zip(server_model.parameters(), model.parameters()):
        proximal_term += (w - w_t).norm(2)
    loss_train += 0.01 / 2. * proximal_term#0.005
    # print(loss_train)
    loss_train.backward()
    optimizer.step()

def train_with_proto(encoder, scorer, optimizer_encoder, optimizer_scorer, features, adj, degrees, labels, class_selected, id_support, id_query, n_way, k_shot):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    loss_train = F.nll_loss(output, labels_new)

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    # [newly added] deduplicate
    query_embeddings = embeddings[list(set(id_query))]
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[list(set(id_query))]])

    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train

def calculate_prototype(encoder, scorer, data, class_list, n_way, degrees):
    prototype_embeddings = dict()
    encoder.eval()
    scorer.eval()
    embeddings = encoder(data.x, data.adj)
    z_dim = embeddings.size()[1]
    scores = scorer(data.x, data.adj)

    for i in range(n_way):
        id_support = class_list[i]
        support_embeddings = embeddings[id_support]
        support_embeddings = support_embeddings.view([1, len(id_support), z_dim])
        support_degrees = torch.log(degrees[id_support]).view([1, len(id_support)])
        support_scores = scores[id_support].view([1, len(id_support)])
        support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
        support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
        support_embeddings = support_embeddings * support_scores
        prototype_embeddings[i] = support_embeddings.sum(1)

    return prototype_embeddings

def train_with_proto_new(encoder, scorer, optimizer_encoder, optimizer_scorer, features, adj, degrees, labels, class_selected, id_support, id_query, n_way, k_shot, global_proto):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    loss_train = F.nll_loss(output, labels_new)

    if global_proto != None:
        global_proto_embedding = dict()
        for key in global_proto.keys():
            global_proto_embedding[key] = copy.deepcopy(global_proto[key].data)
        proto_new = copy.deepcopy(query_embeddings.data)
        for i in range(len(id_query)):
            proto_new[i] = global_proto_embedding[int(labels[id_query[i]])]
        loss_mse = nn.MSELoss()
        loss_train_plus = loss_mse(proto_new, query_embeddings)
        loss_train += 1*loss_train_plus

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    # [newly added] deduplicate
    query_embeddings = embeddings[list(set(id_query))]
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[list(set(id_query))]])

    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train



def train_with_teacher(encoder, scorer, optimizer_encoder, optimizer_scorer, features, adj, degrees, labels, class_selected, id_support, id_query, n_way, k_shot, encoder_t, scorer_t, lam):
    encoder.train()
    scorer.train()
    optimizer_encoder.zero_grad()
    optimizer_scorer.zero_grad()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])
    loss_train = F.nll_loss(output, labels_new)

    loss_train.backward()
    optimizer_encoder.step()
    optimizer_scorer.step()

    # [newly added] deduplicate
    query_embeddings = embeddings[list(set(id_query))]
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)
    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[list(set(id_query))]])

    acc_train = accuracy(output, labels_new)
    f1_train = f1(output, labels_new)

    return acc_train, f1_train


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def eval_test_with_proto(encoder, scorer, features, adj, degrees, labels, class_selected, id_support, id_query, n_way, k_shot):
    encoder.eval()
    scorer.eval()
    embeddings = encoder(features, adj)
    z_dim = embeddings.size()[1]
    scores = scorer(features, adj)

    # embedding lookup
    support_embeddings = embeddings[id_support]
    support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
    query_embeddings = embeddings[id_query]

    # node importance
    support_degrees = torch.log(degrees[id_support].view([n_way, k_shot]))
    support_scores = scores[id_support].view([n_way, k_shot])
    support_scores = torch.sigmoid(support_degrees * support_scores).unsqueeze(-1)
    support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)
    support_embeddings = support_embeddings * support_scores

    # compute loss
    prototype_embeddings = support_embeddings.sum(1)
    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])

    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test

def eval_test(encoder, features, adj, labels, id_query):
    encoder.eval()
    embeddings = encoder(features, adj)
    query_embeddings = embeddings[id_query]
    max_index = torch.argmax(query_embeddings, dim=1)
    labels_new = labels[id_query]
    correct = 0
    for i in range(len(max_index)):
        if max_index[i] == labels_new[i]:
            correct += 1
    acc_test = correct/len(max_index)
    f1 = f1_score(labels_new, max_index, average='weighted')
    return acc_test, f1


def eval_test_with_proto_new(encoder, features, adj, labels, class_selected, id_query, prototype_embeddings):
    encoder.eval()
    embeddings = encoder(features, adj)
    query_embeddings = embeddings[id_query]

    dists = euclidean_dist(query_embeddings, prototype_embeddings)
    output = F.log_softmax(-dists, dim=1)

    labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]])

    acc_test = accuracy(output, labels_new)
    f1_test = f1(output, labels_new)

    return acc_test, f1_test
