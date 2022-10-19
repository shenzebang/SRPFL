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
from utils.test_utils import test_MAML
from models.Update import LocalUpdateHFMAML
from tqdm import tqdm, trange
import time
import ray
import os

@ray.remote(num_gpus=.16)
def ray_dispatch(local, net):
    return local_update(local, net)

def local_update(local, net):
    w_local, loss = local.train(net=net)
    return w_local, loss



if __name__ == '__main__':
    # parse args
    args = args_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


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

    # training
    loss_train = []
    global_accs = [] # records of global model accuracy
    global_accs10 = 0
    start = time.time()


    running_time_record = []
    running_time_all = 0

    # print(torch.cuda.device_count())
    ray.init()
    # print(torch.cuda.device_count())
    for iter in trange(args.epochs):

        set_seed(seeds[iter])

        test_flag = iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 3


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

        users_pool = np.arange(args.num_users)
        idxs_users = np.random.choice(users_pool, max(1, int(args.frac * args.num_users)), replace=False)

        running_time_all += max(simulated_running_time[idxs_users])

        if test_flag:
            running_time_record.append(running_time_all)


        # times_in = []
        total_len = sum([lens[idx] for idx in idxs_users])
        w_glob = {} # accumulates the sum of w_locals


        net_locals = [copy.deepcopy(net_glob).to(args.device) for idx in idxs_users]

        locals = []
        for idx in idxs_users:
            _dataset_train = dataset_train[
                list(dataset_train.keys())[idx][:args.m_tr]] if 'femnist' in args.dataset else dataset_train
            locals.append(LocalUpdateHFMAML(args=args, dataset=_dataset_train, idxs=dict_users_train[idx]))

        results = ray.get([ray_dispatch.remote(local, net_local) for local, net_local in zip(locals, net_locals)])
        w_locals = [result[0] for result in results]
        loss_locals = [result[1] for result in results]

        for w_local, idx in zip(w_locals, idxs_users):
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key]*lens[idx]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_local[key]*lens[idx]


        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # decide if we should double the number of clients in the pool

        # get weighted average for global weights
        for k in net_glob.state_dict().keys():
            w_glob[k] = torch.div(w_glob[k], total_len)


        net_glob.load_state_dict(w_glob)


        # ============= TEST ==================

        if test_flag:

            global_acc_test, loss_test = test_MAML(net_glob, args, dataset_test, dict_users_test,
                                                        dataset_train=dataset_train, dict_users_train=dict_users_train)

            print(f'Testing accuracy: {global_acc_test}')
            global_accs.append(global_acc_test)



    times = np.array(running_time_record)
    save_dir = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.frac == 1:
        global_save_file = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}/HFMAML-{args.description}-global-{args.repeat_id}-{args.hyper_setting}.csv"
    else:
        global_save_file = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}/HFMAML-partial-{args.description}-global-{args.repeat_id}-{args.hyper_setting}.csv"
    global_accs = np.array(global_accs)
    global_accs = pd.DataFrame(np.stack([times, global_accs], axis=1), columns=['times', 'accs'])
    global_accs.to_csv(global_save_file, index=False)