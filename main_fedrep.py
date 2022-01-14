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
from utils.test_utils import test_fine_tune
from models.Update import LocalUpdate
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


    lens = np.ones(args.num_users)
    if 'cifar' in args.dataset or args.dataset == 'mnist':
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

    print(args.alg)

    # build model
    net_glob = get_model(args)
    net_glob.train()
    if args.load_fed != 'n':
        fed_model_path = './save/' + args.load_fed + '.pt'
        net_glob.load_state_dict(torch.load(fed_model_path))

    total_num_layers = len(net_glob.state_dict().keys())
    print(net_glob.state_dict().keys())
    net_keys = [*net_glob.state_dict().keys()]

    # specify the representation parameters (in representation_keys) and head parameters (all others)
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'fedavg' or args.alg == 'prox':
        if 'cifar' in  args.dataset:
            representation_keys = [net_glob.weight_keys[i] for i in [0,1,3,4]]
        elif 'mnist' in args.dataset:
            representation_keys = [net_glob.weight_keys[i] for i in [0,1,2]]
        elif 'sent140' in args.dataset:
            representation_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
        else:
            representation_keys = net_keys[:-2]
    elif args.alg == 'lg':
        if 'cifar' in  args.dataset:
            representation_keys = [net_glob.weight_keys[i] for i in [1,2]]
        elif 'mnist' in args.dataset:
            representation_keys = [net_glob.weight_keys[i] for i in [2,3]]
        elif 'sent140' in args.dataset:
            representation_keys = [net_keys[i] for i in [0,6,7]]
        else:
            representation_keys = net_keys[total_num_layers - 2:]
    else:
        raise NotImplementedError

    # if args.alg == 'fedavg' or args.alg == 'prox':
    #     representation_keys = net_glob.weight_keys
    if 'sent140' not in args.dataset:
        representation_keys = list(itertools.chain.from_iterable(representation_keys))
    
    print(total_num_layers)
    print(representation_keys)
    print(net_keys)
    if args.alg == 'fedrep' or args.alg == 'fedper' or args.alg == 'lg' or args.alg == 'fedavg' or args.alg == 'prox':
        num_param_glob = 0
        num_param_local = 0
        for key in net_glob.state_dict().keys():
            num_param_local += net_glob.state_dict()[key].numel()
            print(num_param_local)
            if key in representation_keys:
                num_param_glob += net_glob.state_dict()[key].numel()
        percentage_param = 100 * float(num_param_glob) / num_param_local
        print('# Params: {} (local), {} (global); Percentage {:.2f} ({}/{})'.format(
            num_param_local, num_param_glob, percentage_param, num_param_glob, num_param_local))
    print("learning rate, batch size: {}, {}".format(args.lr, args.local_bs))

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
    if args.hyper_setting == "noniid-hyper":
        exp_hypers = np.random.uniform(low=args.hyper_low, high=args.hyper_high, size=(args.num_users,))
        simulated_running_time = np.squeeze(np.array([np.random.exponential(hyper, 1) for hyper in exp_hypers]))
    elif args.hyper_setting == "iid-hyper":
        simulated_running_time = np.random.exponential(1, args.num_users)
    else:
        raise NotImplementedError

    double_c = args.double_freq
    if args.init_clients < args.frac * args.num_users:
        raise RuntimeError("the initial pool should be larger than args.frac * args.num_users")
    m = args.init_clients # m is the number of clients in the pool
    running_time_record = []
    running_time_all = 0
    for iter in trange(args.epochs):

        set_seed(seeds[iter])

        test_flag = iter % args.test_freq == args.test_freq - 1 or iter >= args.epochs - 10

        w_glob = {}
        loss_locals = []


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
        idxs_users = np.random.choice(users_pool, int(args.frac * args.num_users), replace=False)

        running_time_all += max(simulated_running_time[idxs_users])

        if test_flag:
            running_time_record.append(running_time_all)


        w_keys_epoch = representation_keys
        times_in = []
        total_len=0
        for ind, idx in enumerate(idxs_users):
            start_in = time.time()
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_ft]], idxs=dict_users_train, indd=indd)
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train[list(dataset_train.keys())[idx][:args.m_tr]], idxs=dict_users_train, indd=indd)
            else:
                if args.epochs == iter:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_ft])
                else:
                    local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users_train[idx][:args.m_tr])

            net_local = copy.deepcopy(net_glob)
            w_local = net_local.state_dict()
            if args.alg != 'fedavg' and args.alg != 'prox':
                for k in local_heads[idx].keys():
                    w_local[k] = local_heads[idx][k]
            net_local.load_state_dict(w_local)
            last = iter == args.epochs
            if 'femnist' in args.dataset or 'sent140' in args.dataset:
                w_local, loss, indd = local.train(net=net_local.to(args.device), ind=idx, idx=clients[idx], representation_keys=representation_keys, lr=args.lr,last=last)
            else:
                w_local, loss, indd = local.train(net=net_local.to(args.device), idx=idx, representation_keys=representation_keys, lr=args.lr, last=last)
            loss_locals.append(copy.deepcopy(loss))
            total_len += lens[idx]
            if len(w_glob) == 0:
                w_glob = copy.deepcopy(w_local)
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] = w_glob[key]*lens[idx]
                    if key not in representation_keys:
                        local_heads[idx][key] = w_local[key]
            else:
                for k,key in enumerate(net_glob.state_dict().keys()):
                    w_glob[key] += w_local[key]*lens[idx]
                    if key not in representation_keys:
                        local_heads[idx][key] = w_local[key]

            times_in.append( time.time() - start_in )
        loss_avg = sum(loss_locals) / len(loss_locals)
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
            if times == []:
                times.append(max(times_in))
            else:
                times.append(times[-1] + max(times_in))

            FT_acc_test, loss_test = test_fine_tune(net_glob, args, dataset_test, dict_users_test,
                                                         representation_keys=representation_keys,
                                                         dataset_train=dataset_train, dict_users_train=dict_users_train)
            print('Round {:3d}, Time {:.3f}, FT, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
                iter, running_time_all, loss_avg, loss_test, FT_acc_test))

            FT_accs.append(FT_acc_test)

            if iter >= args.epochs-10:
                FT_accs10 += FT_acc_test/10

            # below prints the global accuracy of the single global model for the relevant algs
            if args.alg == 'fedavg' or args.alg == 'prox':
                global_acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                        indd=indd,dataset_train=dataset_train, dict_users_train=dict_users_train, return_all=False)
                # print('Round {:3d}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                #     iter, loss_avg, loss_test, global_acc_test))

                global_accs.append(global_acc_test)
                if iter >= args.epochs-10:
                    global_accs10 += global_acc_test/10

        if iter % args.save_every==args.save_every-1:
            model_save_path = './save/accs_'+ args.alg + '_' + args.dataset + '_' + str(args.num_users) +'_'+ str(args.shard_per_user) +'_iter' + str(iter)+ '.pt'
            torch.save(net_glob.state_dict(), model_save_path)

    print('Average accuracy final 10 rounds: {}'.format(FT_accs10))
    if args.alg == 'fedavg' or args.alg == 'prox':
        print('Average global accuracy final 10 rounds: {}'.format(global_accs10))
    end = time.time()
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