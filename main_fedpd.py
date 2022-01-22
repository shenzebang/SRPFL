# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data, set_seed
from models.Update import LocalUpdateFedPD
from models.test import test_img_local_all
from utils.test_utils import test_fine_tune
from tqdm import trange

import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    np.random.seed(args.seed)
    seeds = np.random.randint(1000000, size=3)
    set_seed(seeds)

    seeds = np.random.randint(1000000, size=(args.epochs, 3))

    lens = np.ones(args.num_users)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'mnist' or args.dataset == 'emnist':
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
    else:
        if 'femnist' in args.dataset:
            train_path = './leaf-master/data/' + args.dataset + '/data/mytrain'
            test_path = './leaf-master/data/' + args.dataset + '/data/mytest'
        else:
            train_path = './leaf-master/data/' + args.dataset + '/data/train'
            test_path = './leaf-master/data/' + args.dataset + '/data/test'
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

    net_glob = get_model(args).to(args.device)
    net_glob.train()

    state_dict_global = net_glob.state_dict()

    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    indd = None

    if 'cifar' in args.dataset:
        representation_keys = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
    elif 'mnist' in args.dataset:
        representation_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
    else:
        raise NotImplementedError

    # generate a list of local dual variables
    lambda_list = []
    for user in range(args.num_users):
        lambdai = {}
        for k in net_glob.state_dict().keys():
            lambdai[k] = torch.zeros(state_dict_global[k].size()).to(args.device)
        lambda_list.append(lambdai)

    FT_accs = []
    global_accs = []



    if args.hyper_setting == "noniid-hyper":
        exp_hypers = np.random.uniform(low=args.hyper_low, high=args.hyper_high, size=(args.num_users,))
        simulated_running_time = np.squeeze(np.array([np.random.exponential(hyper, 1) for hyper in exp_hypers]))
    elif args.hyper_setting == "iid-hyper":
        simulated_running_time = np.random.exponential(1, args.num_users)
    else:
        raise NotImplementedError

    double_c = args.double_freq
    m = args.init_clients  # m is the number of clients in the pool
    running_time_record = []
    running_time_all = 0

    for iter in trange(args.epochs):
        set_seed(seeds[iter])
        test_flag = iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10


        if args.resample:
            if args.hyper_setting == "iid-hyper":
                # generate samples from expotential distribution
                simulated_running_time = np.random.exponential(1, args.num_users)
            elif args.hyper_setting == "noniid-hyper":
                simulated_running_time = np.squeeze(np.array([np.random.exponential(hyper, 1) for hyper in exp_hypers]))
            else:
                raise NotImplementedError


        running_time_ordering = np.argsort(simulated_running_time)
        users_pool = running_time_ordering[:m]
        idxs_users = np.random.choice(users_pool, min(m, int(args.frac * args.num_users)), replace=False)

        running_time_all += max(simulated_running_time[idxs_users])

        if test_flag:
            running_time_record.append(running_time_all)


        # perform local update
        loss_locals = []
        for ind, idx in enumerate(idxs_users):

            local = LocalUpdateFedPD(args=args, dataset=dataset_train, idxs=dict_users_train[idx])

            state_dict_local, updated_lambda, loss = local.train(net=net_glob, lambdai=lambda_list[idx])

            loss_locals.append(loss)


            lambda_list[idx] = updated_lambda

            # update accumulate the models
            if ind == 0:
                state_dict_global = state_dict_local
            else:
                for key in state_dict_global.keys():
                    state_dict_global[key] += state_dict_local[key]

        # copy weight to net_glob 
        for k in state_dict_global.keys():
            state_dict_global[k] = torch.div(state_dict_global[k], len(idxs_users))

        net_glob.load_state_dict(state_dict_global)

        loss_avg = sum(loss_locals) / len(loss_locals)

        # decide if we should double the number of clients in the pool
        m = min(m * 2, args.num_users) if double_c == 1 else m
        double_c = args.double_freq if double_c == 1 else double_c - 1

        if test_flag:

            FT_acc_test, loss_test = test_fine_tune(net_glob, args, dataset_test, dict_users_test,
                                                         representation_keys=representation_keys,
                                                         dataset_train=dataset_train, dict_users_train=dict_users_train)
            print('Round {:3d}, Time {:.3f}, FT, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, running_time_all, loss_avg, loss_test, FT_acc_test))

            FT_accs.append(FT_acc_test)



            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox' or args.alg == 'fedpd':
                global_acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                # print('Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                #     iter, loss_avg, loss_test, global_acc_test))

                global_accs.append(global_acc_test)

    times = np.array(running_time_record)

    FT_save_file = f"./save/result-{args.dataset}-{args.shard_per_user}-{args.num_users}-{args.description}-FT-{args.repeat_id}-{args.hyper_setting}.csv"
    FT_accs = np.array(FT_accs)
    FT_accs = pd.DataFrame(np.stack([times, FT_accs], axis=1), columns=['times', 'accs'])
    FT_accs.to_csv(FT_save_file, index=False)

    if args.alg == 'fedavg' or args.alg == 'prox':
        global_save_file = f"./save/result-{args.dataset}-{args.shard_per_user}-{args.num_users}-{args.description}-global-{args.repeat_id}-{args.hyper_setting}.csv"
        global_accs = np.array(global_accs)
        global_accs = pd.DataFrame(np.stack([times, global_accs], axis=1), columns=['times', 'accs'])
        global_accs.to_csv(global_save_file, index=False)

