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
from utils.common_utils import save_results, load_dataset, resample_simulated_running_time, get_simulated_running_time
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
    simulated_running_time = get_simulated_running_time(args)

    np.random.seed(args.seed)
    seeds = np.random.randint(1000000, size=3)
    set_seed(seeds)

    seeds = np.random.randint(1000000, size=(args.epochs, 3))


    dataset_train, dataset_test, dict_users_train, dict_users_test, n_local_datapoints = load_dataset(args)

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
    if args.ray_train: ray.init()
    # print(torch.cuda.device_count())
    for iter in trange(args.epochs):

        set_seed(seeds[iter])

        test_flag = iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 3

        simulated_running_time = resample_simulated_running_time(args, simulated_running_time)

        users_pool = np.arange(args.num_users)
        idxs_users = np.random.choice(users_pool, max(1, int(args.frac * args.num_users)), replace=False)

        running_time_all += max(simulated_running_time[idxs_users])

        if test_flag:
            running_time_record.append(running_time_all)


        # times_in = []
        total_len = sum([n_local_datapoints[idx] for idx in idxs_users])
        w_glob = {} # accumulates the sum of w_locals


        net_locals = [copy.deepcopy(net_glob).to(args.device) for idx in idxs_users]

        locals = []
        for idx in idxs_users:
            _dataset_train = dataset_train[
                list(dataset_train.keys())[idx][:args.m_tr]] if 'femnist' in args.dataset else dataset_train
            locals.append(LocalUpdateHFMAML(args=args, dataset=_dataset_train, idxs=dict_users_train[idx]))

        # results = ray.get([ray_dispatch.remote(local, net_local) for local, net_local in zip(locals, net_locals)])
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
            else:
                for key in net_glob.state_dict().keys():
                    w_glob[key] += w_local[key] * n_local_datapoints[idx]


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

    results = {
        "running_time_record": running_time_record,
        "global_accs": global_accs,
    }

    save_results(args, "HFMAML", results)