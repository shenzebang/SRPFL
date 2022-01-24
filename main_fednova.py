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
from utils.train_utils import get_data, get_model, set_seed
from utils.test_utils import test_fine_tune
from models.Update import LocalUpdateNova
from models.test import test_img_local_all
from tqdm import tqdm, trange
import time

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')


    # control the seed for reproducibility
    np.random.seed(args.seed)
    seeds = np.random.randint(1000000, size=3)
    set_seed(seeds)

    seeds = np.random.randint(1000000, size=(args.epochs, 3))


    dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
    for idx in dict_users_train.keys():
        np.random.shuffle(dict_users_train[idx])


    # build model
    net_global = get_model(args).to(args.device)
    net_global.train()

    if 'cifar' in args.dataset:
        representation_keys = [net_global.weight_keys[i] for i in [0, 1, 3, 4]]
    elif 'mnist' in args.dataset:
        representation_keys = [net_global.weight_keys[i] for i in [0, 1, 2]]
    else:
        raise NotImplementedError

    # training
    indd = None      # indices of embedding for sent140
    loss_train = []
    FT_accs = [] # records of fine tuning model accuracy
    global_accs = [] # records of global model accuracy
    times = []
    FT_accs10 = 0
    global_accs10 = 0
    start = time.time()
    if args.hyper_setting == "noniid-hyper":
        exp_hypers = np.random.uniform(low=args.hyper_low, high=args.hyper_high, size=(args.num_users,))
        simulated_running_time = np.squeeze(np.array([np.random.exponential(hyper, 1) for hyper in exp_hypers]))
    elif args.hyper_setting == "iid-hyper":
        simulated_running_time = np.random.exponential(1, args.num_users)
    else:
        raise NotImplementedError

    double_c = args.double_freq
    m = args.init_clients # m is the number of clients in the pool
    running_time_record = []
    running_time_all = 0
    for iter in trange(args.epochs):

        set_seed(seeds[iter])

        test_flag = iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10

        model_delta_global = {}
        loss_locals = []


        if args.resample:
            if args.hyper_setting == "iid-hyper":
                # generate samples from expotential distribution
                simulated_running_time = np.random.exponential(1, args.num_users)
            elif args.hyper_setting == "noniid-hyper":
                simulated_running_time = np.squeeze(np.array([np.random.exponential(hyper, 1) for hyper in exp_hypers]))
            else:
                raise NotImplementedError

        idxs_users = np.random.choice(args.num_users, min(m, int(args.frac * args.num_users)), replace=False)

        running_time_all += max(simulated_running_time)

        if test_flag:
            running_time_record.append(running_time_all)



        local_eps = np.minimum(10, (max(simulated_running_time) /simulated_running_time).astype(int)) * args.local_ep
        user_weights = np.true_divide(np.sum(local_eps[idxs_users]), local_eps) / len(idxs_users) ** 2 # only the selected users are considered in the reweighting process
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()

            local = LocalUpdateNova(args=args, dataset=dataset_train, idxs=dict_users_train[idx])

            model_delta_local, loss, indd = local.train(net_global=net_global, local_ep=local_eps[idx], lr=args.lr)
            loss_locals.append(loss)

            if len(model_delta_global) == 0:
                model_delta_global = model_delta_local
                for key in model_delta_local.keys():
                    model_delta_global[key] *= user_weights[idx]
            else:
                for key in model_delta_local.keys():
                    model_delta_global[key] += model_delta_local[key]*user_weights[idx]



        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)

        # decide if we should double the number of clients in the pool
        m = min(m * 2, args.num_users) if double_c == 1 else m
        double_c = args.double_freq if double_c == 1 else double_c - 1

        sd = net_global.state_dict()
        for key in model_delta_global.keys():
            sd[key] += model_delta_global[key]

        net_global.load_state_dict(sd)

        if test_flag:
            FT_acc_test, loss_test = test_fine_tune(net_global, args, dataset_test, dict_users_test,
                                                         representation_keys=representation_keys,
                                                         dataset_train=dataset_train, dict_users_train=dict_users_train)
            print('Round {:3d}, Time {:.3f}, FT, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, running_time_all, loss_avg, loss_test, FT_acc_test))

            FT_accs.append(FT_acc_test)

            if iter >= args.epochs-10:
                FT_accs10 += FT_acc_test/10

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                global_acc_test, loss_test = test_img_local_all(net_global, args, dataset_test, dict_users_test,
                                                        indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                # print('Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                #     iter, loss_avg, loss_test, global_acc_test))

                global_accs.append(global_acc_test)
                if iter >= args.epochs-10:
                    global_accs10 += global_acc_test/10

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_global.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(FT_accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(global_accs10))
    # print(end-start)
    # print(times)
    # print(accs)
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