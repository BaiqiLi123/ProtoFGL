import json
import os
import argparse
from copy import deepcopy
# from torch.utils.tensorboard import SummaryWriter
from config import *
from Preprocess.data_process import *
from Model.models import *
from fedavg import *
from utils import *
# Dataset
from utils import _task_generator


# writer = SummaryWriter("summary")


def run_test_with_proto(_encoder, _scorer):
    # testing
    _meta_test_acc = []
    _meta_test_f1 = []
    for idx in range(meta_test_num):
        id_support, id_query, class_selected = test_pool[idx]
        acc_test, f1_test = eval_test_with_proto(_encoder, _scorer, global_data.x, global_data.adj, \
                                                 global_degree, global_data.y, \
                                                 class_selected, id_support, id_query, args.n_way, args.k_shot_test)
        _meta_test_acc.append(acc_test)
        _meta_test_f1.append(f1_test)

    return np.array(_meta_test_acc).mean(axis=0), np.array(_meta_test_f1).mean(axis=0)


def run_test_with_proto_new(_encoder, prototype):
    # testing
    _meta_test_acc = []
    _meta_test_f1 = []
    for idx in range(meta_test_num):
        id_support, id_query, class_selected = test_pool[idx]
        acc_test, f1_test = eval_test_with_proto_new(_encoder, global_data.x, global_data.adj, \
                                                     global_data.y, \
                                                     class_selected, id_query, prototype)
        _meta_test_acc.append(acc_test)
        _meta_test_f1.append(f1_test)

    return np.array(_meta_test_acc).mean(axis=0), np.array(_meta_test_f1).mean(axis=0)


def run_test(base):
    # testing
    _meta_test_acc = []
    _meta_test_f1 = []
    for idx in range(meta_test_num):
        id_support, id_query, class_selected = test_pool[idx]
        acc_test, f1_test = eval_test(base, global_data.x, global_data.adj, \
                                      global_data.y, id_query)
        _meta_test_acc.append(acc_test)
        _meta_test_f1.append(f1_test)
    return np.array(_meta_test_acc).mean(axis=0), np.array(_meta_test_f1).mean(axis=0)


def run_local(_base, _optimizer_base, data, epochs, if_test=True):
    meta_train_acc = []
    meta_train_f1 = []
    meta_test_acc = []
    meta_test_f1 = []

    for epoch in range(epochs):
        id_support, id_query = _task_generator(len(data.y), args.k_shot * args.n_way, args.n_query * args.n_way)
        train(_base, _optimizer_base, data.x, data.adj, data.y, id_support, id_query)

        if epoch > 0 and epoch % 10 == 0 and if_test == True:
            print("-------Epoch {}-------".format(epoch))
            # print("Meta-Train_Accuracy: {}, Meta-Train_F1: {}".format(acc_train, f1_train))
            # testing
            test_result = run_test(_base)
            meta_test_acc.append(test_result[0])
            meta_test_f1.append(test_result[1])
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(test_result[0], test_result[1]))
    test_result = run_test(_base)
    last_acc = test_result[0]
    last_F1 = test_result[1]

    return meta_train_acc, meta_train_f1, meta_test_acc, meta_test_f1, last_acc, last_F1


def run_local_prox(_base, _optimizer_base, _server_base, data, epochs, if_test=True):
    meta_train_acc = []
    meta_train_f1 = []
    meta_test_acc = []
    meta_test_f1 = []
    for epoch in range(epochs):
        id_support, id_query = _task_generator(len(data.y), args.k_shot * args.n_way, args.n_query * args.n_way)
        train_prox(_base, _optimizer_base, _server_base, data.x, data.adj, data.y, id_support, id_query)
        if epoch > 0 and epoch % 10 == 0 and if_test == True:
            print("-------Epoch {}-------".format(epoch))
            # print("Meta-Train_Accuracy: {}, Meta-Train_F1: {}".format(acc_train, f1_train))
            # testing
            test_result = run_test(_base)
            meta_test_acc.append(test_result[0])
            meta_test_f1.append(test_result[1])
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(test_result[0], test_result[1]))
    return meta_train_acc, meta_train_f1, meta_test_acc, meta_test_f1


def run_local_with_proto(_encoder, _scorer, _optimizer_encoder, _optimizer_scorer, _degree, data, _class_list, epochs,
                         if_test=True):
    meta_train_acc = []
    meta_train_f1 = []
    meta_test_acc = []
    meta_test_f1 = []

    train_class = []
    for i in _class_list.keys():
        if len(_class_list[i]) != 0:
            train_class.append(i)

    for epoch in range(epochs):
        id_support, id_query, class_selected = \
            task_generator(_class_list, train_class, args.n_way, args.k_shot, args.n_query)

        acc_train, f1_train = train_with_proto(_encoder, _scorer, _optimizer_encoder, _optimizer_scorer, data.x,
                                               data.adj, _degree, data.y, class_selected,
                                               id_support, id_query, len(train_class), args.k_shot)

        meta_train_acc.append(acc_train)
        meta_train_f1.append(f1_train)

        if epoch > 0 and epoch % 10 == 0 and if_test == True:
            print("-------Epoch {}-------".format(epoch))
            # print("Meta-Train_Accuracy: {}, Meta-Train_F1: {}".format(acc_train, f1_train))
            # testing
            test_result = run_test_with_proto(_encoder, _scorer)
            meta_test_acc.append(test_result[0])
            meta_test_f1.append(test_result[1])
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(test_result[0], test_result[1]))

    return meta_train_acc, meta_train_f1, meta_test_acc, meta_test_f1


def main():

    if args.mode == 'global':  # gnn-only
        _meta_train_acc, _meta_train_f1, _meta_test_acc, _meta_test_f1, \
            _last_acc, _last_F1 = run_local(_base=base,
                                            _optimizer_base=optimizer_base,
                                            data=global_data,
                                            epochs=args.iters,
                                            if_test=True)
        # test_result = run_test(base)
        # print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(test_result[0], test_result[1]))
        # summary["msg"].append({"Meta-Test_Accuracy": _meta_test_acc, "Meta-Test_F1": _meta_test_f1})
        summary["acc"] = {"last": _last_acc}
        summary["F1"] = {"last": _last_F1}

    elif args.mode == 'FedAvg_with_proto_new':  # ProtoFGL
        alg = fedavg(args, encoder, scorer)
        for round in range(args.global_rounds):
            for idx in range(args.n_clients):
                # print("-------Client {}-------".format(idx))
                alg.client_prototype[idx] = \
                    run_local_with_proto_new(_encoder=alg.client_model[idx]['encoder'],
                                             _scorer=alg.client_model[idx]['scorer'], \
                                             _optimizer_encoder=alg.optimizer_encoders[idx], \
                                             _optimizer_scorer=alg.optimizer_scorers[idx], _degree=client_degree[idx],
                                             data=clients_data[idx], \
                                             _class_list=client_class_list[idx], epochs=args.local_iters, if_test=False,
                                             _global_proto=alg.server_prototype)
            alg.aggregate_prototype(client_class_list, args.hidden, args.n_way)
            alg.aggregate()
            print("-------Round {}-------".format(round))
            test_result = run_test_with_proto(alg.server_model['encoder'], alg.server_model['scorer'])
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(test_result[0], test_result[1]))
            summary["msg"].append(
                {"Round": round, "Meta-Test_Accuracy": test_result[0], "Meta-Test_F1": test_result[1]})
        _accmax = 0
        _acc_maxclient = None

        _F1max = 0
        _F1_maxclient = None

        for ii in summary["msg"]:
            if ii["Meta-Test_Accuracy"] > _accmax:
                _accmax = ii["Meta-Test_Accuracy"]
                _acc_maxclient = deepcopy(ii)
            if ii["Meta-Test_F1"] > _F1max:
                _F1max = ii["Meta-Test_F1"]
                _F1_maxclient = deepcopy(ii)

        summary["acc"] = {"max": _acc_maxclient}
        summary["F1"] = {"max": _F1_maxclient}
    # filedir = 'LDA'+str(args.alpha)+' 50501010 5_200/' + args.dataset + "/"
    filedir = "three_pubmed/"
    filename = str(time.time()) +args.dataset+args.mode + "_" + str(args.alpha) + "_" + "client" + str(args.n_clients) + "_" + '.json'
    os.makedirs(filedir, exist_ok=True)
    with open(filedir + filename, 'w') as f:
        json.dump(summary, f, indent=4)


def parse_args():
    description = "..."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-d", "--dataset", type=str, help="client_num")
    parser.add_argument("-c", "--client", type=int, help="client_num")
    parser.add_argument("-m", "--mode", type=str, help="algorithm")
    parser.add_argument("-s", "--splitter", type=str, help="split_ways")
    parser.add_argument("-a", "--alpha", type=float, default=-1, help="LDA_alpha")

    return parser.parse_args()


if __name__ == '__main__':
    args_input = parse_args()
    data_way = {"cora": 7, "citeseer": 6, "pubmed": 3}
    args = ARGS(dataset=args_input.dataset, n_way=data_way[args_input.dataset], splitter=args_input.splitter,
                n_clients=args_input.client, seed=123, iters=1000, threshold=0.8, lam=0.1,
                lr=0.1)

    # args.mode = 'FedAvg_with_proto_new'
    args.mode = args_input.mode
    args.alpha = args_input.alpha

    setup_seed(args.seed)
    global_data, clients_data = load_data(args.dataset, args.splitter, args.file_path, args.n_clients, alpha=args.alpha)
    # analyze_datasets(clients_data)
    device = 'cpu'
    # [Newly add] data preprocess
    # args.n_way = 6
    args.k_shot_test = 50  # 50
    args.n_query_test = 50  # 50
    args.k_shot = 10
    args.n_query = 10
    args.local_iters = 5  # 30
    args.global_rounds = 200  # 3000 50*600=3000
    args.lam = 10

    if args.mode == 'local' or args.mode == 'local_with_proto':
        summary = {"dataset": args.dataset, "mode": args.mode, "n_clients": args.n_clients, "iter": args.iters,
                   "args.k_shot_test": args.k_shot_test, "args.n_query_test": args.n_query_test,
                   "args.k_shot ": args.k_shot, "args.n_query ": args.n_query,

                   "splitter": args.splitter, "alpha": args.alpha, "lr": args.lr,
                   "acc": None, "F1": None, "msg": []}
    else:
        summary = {"dataset": args.dataset, "mode": args.mode, "n_clients": args.n_clients,
                   "global_rounds": args.global_rounds,
                   "local_iters": args.local_iters,
                   "args.k_shot_test": args.k_shot_test, "args.n_query_test": args.n_query_test,
                   "args.k_shot ": args.k_shot, "args.n_query ": args.n_query,

                   "splitter": args.splitter, "alpha": args.alpha, "lr": args.lr,
                   "acc": None, "F1": None, "msg": []}

    global_class_list, client_class_list = class_list_generator(global_data, clients_data, args)
    global_degree = torch.sum(global_data.adj, axis=1).reshape((-1, 1))
    global_data.adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(global_data.adj))
    client_degree = []
    for i in range(args.n_clients):
        # print(clients_data[i].adj)
        client_degree.append(torch.sum(clients_data[i].adj, axis=1).reshape((-1, 1)))
        clients_data[i].adj = sparse_mx_to_torch_sparse_tensor(normalize_adj(clients_data[i].adj))

    # [Newly add] Model and optimizer
    encoder = Encoder(nfeat=global_data.x.shape[1],
                          nhid=args.hidden,
                          dropout=args.dropout)
    scorer = Valuator(nfeat=global_data.x.shape[1],
                          nhid=args.hidden,
                          dropout=args.dropout)
    optimizer_encoder = optim.Adam(encoder.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    optimizer_scorer = optim.Adam(scorer.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)

    base = Base(nfeat=global_data.x.shape[1],
                    nhid=args.hidden,
                    dropout=args.dropout,
                    nclass=args.n_way)
    optimizer_base = optim.Adam(base.parameters(),
                                lr=args.lr, weight_decay=args.weight_decay)

    # [Newly add] Test task
    meta_test_num = 50
    test_class = [i for i in range(args.n_way)]
    test_pool = [task_generator(global_class_list, test_class, args.n_way, args.k_shot_test, args.n_query_test) for i in
                 range(meta_test_num)]

    main()
