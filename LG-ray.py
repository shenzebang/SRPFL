# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

# This program implements FedRep under the specification --alg fedrep, as well as Fed-Per (--alg fedper), LG-FedAvg (--alg lg), 
# FedAvg (--alg fedavg) and FedProx (--alg prox)

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data, set_seed
from utils.test_utils import test_fine_tune, test_fine_tune_ray
from models.Update import LocalUpdateLG
from models.test import test_img_local_all
from tqdm import tqdm, trange
import time, os

import ray

@ray.remote(num_gpus=.14)
def ray_dispatch(local, net):
    return local.train(net=net)



if __name__ == '__main__':
    # parse args
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # control the seed for reproducibility
    np.random.seed(1)
    if args.hyper_setting == "noniid-hyper":
        exp_hypers = np.random.uniform(low=args.hyper_low, high=args.hyper_high, size=(args.num_users,))
        simulated_running_time = np.squeeze(np.array([np.random.exponential(hyper, 1) for hyper in exp_hypers]))
    elif args.hyper_setting == "iid-hyper":
        simulated_running_time = np.random.exponential(1, args.num_users)
        # This is added only for the purpose of ICML rebuttal
        if args.reserve:
            simulated_running_time = np.sort(simulated_running_time)
            simulated_running_time_not_reserved = simulated_running_time[:int(0.8 * args.num_users)]
            simulated_running_time_reserved = simulated_running_time[int(0.8 * args.num_users):]
            np.random.shuffle(simulated_running_time_not_reserved)
            np.random.shuffle(simulated_running_time_reserved)
            simulated_running_time = np.concatenate(
                [simulated_running_time_not_reserved, simulated_running_time_reserved])
    else:
        raise NotImplementedError
    np.random.seed(args.seed)
    seeds = np.random.randint(1000000, size=3)
    set_seed(seeds)

    seeds = np.random.randint(1000000, size=(args.epochs, 3))


    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist' or args.dataset == 'emnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        if 'femnist' in args.dataset:
            train_path = f'data/femnist/mytrain'
            test_path = f'data/femnist/mytest'
        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        lens = []
        for iii, c in enumerate(clients):
            lens.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        print(lens)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    # build model
    net_glob = get_model(args)
    net_glob.train()

    total_num_layers = len(net_glob.state_dict().keys())
    # print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in representation_keys) and head parameters (all others)


    if 'cifar' in  args.dataset:
        representation_keys = [net_glob.weight_keys[i] for i in [1,2]]
    elif 'mnist' in args.dataset:
        representation_keys = [net_glob.weight_keys[i] for i in [2,3]]
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


    ray.init()

    for iter in trange(args.epochs):

        set_seed(seeds[iter])

        test_flag = iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 3

        w_glob = {}
        loss_locals = []

        if args.hyper_setting == "iid-hyper":
            if args.resample:
                # regenerate samples from expotential distribution
                simulated_running_time = np.random.exponential(1, args.num_users)
        elif args.hyper_setting == "noniid-hyper":
            if args.resample:
                exp_hypers = np.random.uniform(low=args.hyper_low, high=args.hyper_high, size=(args.num_users,))
            simulated_running_time = np.squeeze(np.array([np.random.exponential(hyper, 1) for hyper in exp_hypers]))
        else:
            raise NotImplementedError

        if args.flanp:
            active_users_pool = np.random.choice(args.num_users, max(1, int(args.frac * args.num_users)), replace=False)
            simulated_running_time_in_pool = simulated_running_time[active_users_pool]
            running_time_ordering = np.argsort(simulated_running_time_in_pool)
            users_pool = running_time_ordering[:m]
            idxs_users = active_users_pool[users_pool]
        else:
            users_pool = np.arange(args.num_users)
            idxs_users = np.random.choice(users_pool, max(1, int(args.frac * args.num_users)), replace=False)

        running_time_all += max(simulated_running_time[idxs_users])

        if test_flag:
            running_time_record.append(running_time_all)



        total_len=0

        net_locals = [copy.deepcopy(net_glob).to(args.device) for idx in idxs_users]

        for net_local, idx in zip(net_locals, idxs_users):
            w_local = net_local.state_dict()
            for k in local_heads[idx].keys():
                w_local[k] = local_heads[idx][k]
            net_local.load_state_dict(w_local)

        locals = []
        for idx in idxs_users:
            _dataset_train = dataset_train[
                list(dataset_train.keys())[idx][:args.m_tr]] if 'femnist' in args.dataset else dataset_train
            locals.append(LocalUpdateLG(args=args, dataset=_dataset_train, idxs=dict_users_train[idx],
                                            representation_keys=representation_keys))

        results = ray.get([ray_dispatch.remote(local, net_local)
                           for local, net_local in zip(locals, net_locals)])
        w_locals = [result[0] for result in results]
        loss_locals = [result[1] for result in results]


        for w_local, idx in zip(w_locals, idxs_users):
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for key in net_glob.state_dict().keys():
                    if key not in representation_keys:
                        local_heads[idx][key] = w_local[key]
            else:
                for key in net_glob.state_dict().keys():
                    w_glob[key] += w_local[key]
                    if key not in representation_keys:
                        local_heads[idx][key] = w_local[key]

        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # decide if we should double the number of clients in the pool
        m = min(m * 2, args.num_users) if double_c == 1 else m
        double_c = args.double_freq if double_c == 1 else double_c - 1

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], len(idxs_users))

        w_local = net_glob.state_dict()
        for k in w_glob.keys():
            w_local[k] = w_glob[k]

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
    times = np.array(running_time_record)
    save_dir = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.frac == 1:
        FT_save_file = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}/LG-{args.description}-FT-{args.repeat_id}-{args.hyper_setting}.csv"
    else:
        FT_save_file = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}/LG-partial-{args.description}-FT-{args.repeat_id}-{args.hyper_setting}.csv"
    FT_accs = np.array(FT_accs)
    FT_accs = pd.DataFrame(np.stack([times, FT_accs], axis=1), columns=['times', 'accs'])
    FT_accs.to_csv(FT_save_file, index=False)

