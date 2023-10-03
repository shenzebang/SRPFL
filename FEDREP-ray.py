# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)
import os
import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data, set_seed
from utils.test_utils import test_fine_tune, test_fine_tune_ray
from models.Update import LocalUpdateFEDREP
from models.test import test_img_local_all
from tqdm import tqdm, trange
import time

import ray

from utils.common_utils import save_results, load_dataset, get_simulated_running_time, resample_simulated_running_time
@ray.remote(num_gpus=.2)
def ray_dispatch(local, net):
    return local_update(local, net)

def local_update(local, net):
    w_local, loss = local.train(net=net)
    return w_local, loss




if __name__ == '__main__':
    # parse args
    args = args_parser()
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # control the seed for reproducibility
    np.random.seed(1)
    simulated_running_time = get_simulated_running_time(args)

    np.random.seed(args.seed)
    seeds = np.random.randint(1000000, size=3)
    set_seed(seeds)

    seeds = np.random.randint(1000000, size=(args.epochs, 3))


    dataset_train, dataset_test, dict_users_train, dict_users_test, n_local_datapoints = load_dataset(args)

    # build model
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    # print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in representation_keys) and head parameters (all others)

    if 'cifar' in  args.dataset:
        representation_keys = [net_glob.weight_keys[i] for i in [0,1,3,4]]
    elif 'mnist' in args.dataset:
        representation_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
    else:
        raise NotImplementedError

    representation_keys = list(itertools.chain.from_iterable(representation_keys))

    # generate list of local heads for each user
    local_heads = {}
    for user in range(args.num_users):
        _head = {}
        for key in net_glob.state_dict().keys():
            if key not in representation_keys:
                _head[key] = net_glob.state_dict()[key]
        local_heads[user] = _head

    # training
    indd = None      # indices of embedding for sent140
    loss_train = []
    FT_accs = [] # records of fine tuning model accuracy
    global_accs = [] # records of global model accuracy
    times = []
    FT_accs10 = 0
    global_accs10 = 0
    start = time.time()


    double_c = args.double_freq
    m = min(args.num_users, args.init_clients) # m is the number of clients in the pool
    running_time_record = []
    running_time_all = 0


    if args.ray_train: ray.init()

    for iter in trange(args.epochs):

        set_seed(seeds[iter])

        test_flag = iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 3

        w_glob = {}
        loss_locals = []

        simulated_running_time = resample_simulated_running_time(args, simulated_running_time)

        if args.flanp:
            active_users_pool = np.random.choice(args.num_users, max(1, int(args.frac * args.num_users)), replace=False)
            simulated_running_time_in_pool = simulated_running_time[active_users_pool]
            running_time_ordering = np.argsort(simulated_running_time_in_pool)
            users_pool = running_time_ordering[:m]
            idxs_users = active_users_pool[users_pool]
            # idxs_users = np.random.choice(users_pool, min(m, int(args.frac * args.num_users)), replace=False)
        else:
            users_pool = np.arange(args.num_users)
            idxs_users = np.random.choice(users_pool, max(1, int(args.frac * args.num_users)), replace=False)

        running_time_all += max(simulated_running_time[idxs_users])

        if test_flag:
            running_time_record.append(running_time_all)


        total_len = sum([n_local_datapoints[idx] for idx in idxs_users])

        net_locals = [copy.deepcopy(net_glob).to(args.device) for idx in idxs_users]

        for net_local, idx in zip(net_locals, idxs_users):
            w_local = net_local.state_dict()
            for k in local_heads[idx].keys():
                w_local[k] = local_heads[idx][k]
            net_local.load_state_dict(w_local)

        locals = []
        for idx in idxs_users:
            _dataset_train = dataset_train[list(dataset_train.keys())[idx][:args.m_tr]] if 'femnist' in args.dataset else dataset_train
            locals.append(LocalUpdateFEDREP(args=args, dataset=_dataset_train, idxs=dict_users_train[idx], representation_keys=representation_keys))

        if args.ray_train:
            results = ray.get([ray_dispatch.remote(local, net_local) for local, net_local in zip(locals, net_locals)])
        else:
            results = [local_update(local, net_local) for local, net_local in zip(locals, net_locals)]
        w_locals = [result[0] for result in results]
        loss_locals = [result[1] for result in results]


        for w_local, idx in zip(w_locals, idxs_users):
            if len(w_glob) == 0:
                # w_glob = copy.deepcopy(w_local)
                for key in net_glob.state_dict():
                    w_glob[key] = copy.deepcopy(w_local[key]) * n_local_datapoints[idx]
                for key in net_glob.state_dict().keys():
                    if key not in representation_keys:
                        local_heads[idx][key] = w_local[key]
            else:
                for key in net_glob.state_dict().keys():
                    w_glob[key] += w_local[key] * n_local_datapoints[idx]
                    if key not in representation_keys:
                        local_heads[idx][key] = w_local[key]


        loss_avg = sum([loss_local * n_local_datapoints[idx] for idx, loss_local in enumerate(loss_locals)]) / total_len
        loss_train.append(loss_avg)

        # decide if we should double the number of clients in the pool
        m = min(m * 2, args.num_users) if double_c == 1 else m
        double_c = args.double_freq if double_c == 1 else double_c - 1

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)

        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]
        if args.epochs != iter:
            net_glob.load_state_dict(w_glob)

        if test_flag:

            if args.ray_test:
                FT_acc_test, loss_test = test_fine_tune_ray(net_glob, args, dataset_test, dict_users_test,
                                                        representation_keys=representation_keys,
                                                        dataset_train=dataset_train, dict_users_train=dict_users_train)
            else:
                FT_acc_test, loss_test = test_fine_tune(net_glob, args, dataset_test, dict_users_test,
                                                         representation_keys=representation_keys,
                                                         dataset_train=dataset_train, dict_users_train=dict_users_train)
            print('Round {:3d}, Time {:.3f}, FT, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, running_time_all, loss_avg, loss_test, FT_acc_test))

            FT_accs.append(FT_acc_test)

            if iter >= args.epochs-10:
                FT_accs10 += FT_acc_test/10

    print('Average accuracy final 10 rounds: {}'.format(FT_accs10))
    # print(end-start)
    # print(times)
    # print(accs)
    results = {
        "running_time_record": running_time_record,
        "FT_accs": FT_accs,
    }

    save_results(args, "FEDREP", results)
