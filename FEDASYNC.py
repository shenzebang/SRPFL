# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/main_fed.py
# credit goes to: Paul Pu Liang

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import itertools
import numpy as np
import pandas as pd
import torch
from torch import nn

from utils.options import args_parser
from utils.train_utils import get_data, get_model, read_data, set_seed, DatasetSplit
from utils.test_utils import test_fine_tune, test_fine_tune_ray
from models.test import test_img_local_all
from tqdm import tqdm, trange
import time
import os

from utils.scheduling_utils import User, AsynUserPool
from torch.utils.data import DataLoader


class LocalUpdateFEDASYN(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.dataset = dataset
        self.idxs = idxs

    def train(self, net: nn.Module):
        sd_old = copy.deepcopy(net).requires_grad_(False).state_dict()
        def l2_regularization(sd):
            residual = torch.zeros([])
            for key in sd:
                residual = residual + torch.sum((sd[key]-sd_old[key])**2)
            return residual

        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=self.args.lr
        )
        epoch_loss = []
        num_updates = 0

        for name, param in net.named_parameters():
            param.requires_grad = True



        for iter in range(self.args.FEDAVG_local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels) + 0.005 * l2_regularization(net.state_dict())
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

def dispatch(local, net):
    return local_update(local, net)

def local_update(local, net):
    w_local, loss = local.train(net=net)
    return w_local, loss


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
        simulated_running_time = np.random.randint(low=10, high=100, size=(args.num_users,))
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

    if 'cifar' in args.dataset:
        representation_keys = [net_glob.weight_keys[i] for i in [0, 1, 3, 4]]
    elif 'mnist' in args.dataset:
        representation_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
    else:
        raise NotImplementedError

    representation_keys = list(itertools.chain.from_iterable(representation_keys))


    # training
    indd = None  # indices of embedding for sent140
    loss_train = []
    FT_accs = []  # records of fine tuning model accuracy
    global_accs = []  # records of global model accuracy
    # FT_accs10 = 0
    # global_accs10 = 0
    start = time.time()


    double_c = args.double_freq
    m = min(args.num_users, args.init_clients)  # m is the number of clients in the pool
    running_time_record = []
    running_time_all = 0

    # initialize users
    users = [User(uid, copy.deepcopy(net_glob), running_time) for uid, running_time in enumerate(simulated_running_time)]
    # initialize the user pool for asynchronous computation
    user_pool = AsynUserPool(users)

    for system_time in trange(args.maximum_system_time):
        test_flag = system_time % args.test_freq == args.test_freq - 1

        user_ready = user_pool.return_ready()
        if len(user_ready) != 0:
            # results = ray.get([
            #     ray_dispatch.remote(LocalUpdateFEDASYN(args=args, dataset=dataset_train, idxs=dict_users_train[user.uid]),
            #                         user.current_model)
            #     for user in user_ready
            # ])
            results = [dispatch(LocalUpdateFEDASYN(args=args, dataset=dataset_train, idxs=dict_users_train[user.uid]),
                        user.current_model) for user in user_ready]

            loss_locals = [result[1] for result in results]
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)

            w_locals = [result[0] for result in results]
            w_glob = {}
            for w_local in w_locals:
                if len(w_glob) == 0:
                    w_glob = copy.deepcopy(w_local)
                else:
                    for key in net_glob.state_dict().keys():
                        w_glob[key] += w_local[key]
            # get weighted average for global weights
            sd_old = net_glob.state_dict()
            for k in sd_old.keys():
                w_glob[k] = torch.div(w_glob[k], len(user_ready)) * args.global_lr + (1 - args.global_lr) * sd_old[k]
            net_glob.load_state_dict(w_glob)

        user_pool.reset(net_glob)
        user_pool.step()


        if test_flag:
            running_time_record.append(system_time)
            # time_FT_test = time.time()
            # if args.ray_test:
            #     FT_acc_test, loss_test = test_fine_tune_ray(net_glob, args, dataset_test, dict_users_test,
            #                                                 representation_keys=representation_keys,
            #                                                 dataset_train=dataset_train,
            #                                                 dict_users_train=dict_users_train)
            # else:
            #     FT_acc_test, loss_test = test_fine_tune(net_glob, args, dataset_test, dict_users_test,
            #                                             representation_keys=representation_keys,
            #                                             dataset_train=dataset_train, dict_users_train=dict_users_train)
            #
            # time_FT_test = time.time() - time_FT_test
            # print("FT test time is {}".format(time_FT_test))
            #
            # print('Time {:.3f}, FT, Train loss: {:.3f}, Test loss: {:.3f}, Test accuracy: {:.2f}'.format(
            #     system_time, loss_avg, loss_test, FT_acc_test))
            #
            # FT_accs.append(FT_acc_test)

            # below prints the global accuracy of the single global model for the relevant algs
            global_acc_test, loss_test = test_img_local_all(net_glob, args, dataset_test, dict_users_test,
                                                            indd=indd, dataset_train=dataset_train,
                                                            dict_users_train=dict_users_train, return_all=False)
            print('Time {:.3f}, Global train loss: {:.3f}, Global test loss: {:.3f}, Global test accuracy: {:.2f}'.format(
                system_time, loss_avg, loss_test, global_acc_test))

            global_accs.append(global_acc_test)

    # print('Average accuracy final 10 rounds: {}'.format(FT_accs10))
    # print(end-start)
    # print(times)
    # print(accs)
    times = np.array(running_time_record)
    save_dir = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # if args.frac == 1:
    #     FT_save_file = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}/FEDAYNC-{args.description}-FT-{args.repeat_id}-{args.hyper_setting}.csv"
    # else:
    #     FT_save_file = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}/FEDAYNC-partial-{args.description}-FT-{args.repeat_id}-{args.hyper_setting}.csv"
    # FT_accs = np.array(FT_accs)
    # FT_accs = pd.DataFrame(np.stack([times, FT_accs], axis=1), columns=['times', 'accs'])
    # FT_accs.to_csv(FT_save_file, index=False)

    if args.frac == 1:
        global_save_file = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}/FEDASYNC-{args.description}-global-{args.repeat_id}-{args.hyper_setting}.csv"
    else:
        global_save_file = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}/FEDASYNC-partial-{args.description}-global-{args.repeat_id}-{args.hyper_setting}.csv"
    global_accs = np.array(global_accs)
    global_accs = pd.DataFrame(np.stack([times, global_accs], axis=1), columns=['times', 'accs'])
    global_accs.to_csv(global_save_file, index=False)