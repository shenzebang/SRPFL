import numpy as np
import copy
from torch.utils.data import DataLoader
from models.Update import DatasetSplit
import torch
import torch.nn as nn
from models.test import test_img_local
import time

import ray
import os

@ray.remote(num_gpus=.14)
def ray_dispatch(args, net, dataloader, representation_keys):
    return fine_tune(net, args, dataloader, representation_keys)


def test_fine_tune_ray(net, args, dataset_test, dict_users_test, representation_keys, dataset_train, dict_users_train, indd=None):
    tot = 0
    acc_test_local = np.zeros(args.num_users)
    loss_test_local = np.zeros(args.num_users)

    net_locals = [copy.deepcopy(net) for _ in range(args.num_users)]
    dataloaders = [DataLoader(DatasetSplit(dataset_train, dict_users_train[idx]), batch_size=args.local_bs, shuffle=True)
                   for idx in range(args.num_users)]

    time_FT = time.time()
    net_locals = ray.get([ray_dispatch.remote(args, net_local, dataloader, representation_keys)
                          for net_local, dataloader in zip(net_locals, dataloaders)])
    print("time FT is {}".format(time.time() - time_FT))


    for idx, net_local in enumerate(net_locals):
        a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
        tot += len(dict_users_test[idx])
        acc_test_local[idx] = a*len(dict_users_test[idx])
        loss_test_local[idx] = b*len(dict_users_test[idx])



    return sum(acc_test_local) / tot, sum(loss_test_local) / tot


def test_fine_tune(net, args, dataset_test, dict_users_test, representation_keys, dataset_train, dict_users_train, indd=None):
    tot = 0
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    net_local = copy.deepcopy(net)

        # get the keys for the head
    head_keys = []
    for key in net.state_dict().keys():
        if key not in representation_keys:
            head_keys.append(key)

    head_dict = {key: copy.deepcopy(net.state_dict()[key]) for key in head_keys}


    for idx in range(num_idxxs):
        # reset the head
        w_local = net_local.state_dict()
        for key in head_keys:
            w_local[key] = copy.deepcopy(head_dict[key])
        net_local.load_state_dict(w_local)

        # fine tune the head
        net_local.train()
        if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'emnist' or args.dataset == 'mnist':
            dataloader = DataLoader(DatasetSplit(dataset_train, dict_users_train[idx]), batch_size=args.local_bs, shuffle=True)
        else:
            raise NotImplementedError

        fine_tune(net_local, args, dataloader, representation_keys)

        # test

        net_local.eval()


        a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
        tot += len(dict_users_test[idx])


        acc_test_local[idx] = a*len(dict_users_test[idx])
        loss_test_local[idx] = b*len(dict_users_test[idx])



    return sum(acc_test_local) / tot, sum(loss_test_local) / tot

def fine_tune(net, args, dataloader, representation_keys):
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
        lr=args.lr, momentum=0.5
    )

    for name, param in net.named_parameters():
        if name in representation_keys:
            param.requires_grad = False
        else:
            param.requires_grad = True

    loss_func = nn.CrossEntropyLoss()
    for epoch in range(args.FT_epoch):
        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(args.device), labels.to(args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            loss.backward()
            optimizer.step()

    return net




# ====== TEST MODULE OF HF-MAML ======
def test_MAML(net: nn.Module, args, dataset_test, dict_users_test, dataset_train, dict_users_train):
    tot = 0
    loss_func = nn.CrossEntropyLoss()
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)
    sd = net.state_dict()


    for idx in range(num_idxxs):

        sd_idx = copy.copy(sd)

        # take a gradient step

        if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'emnist' or args.dataset == 'mnist':
            dataloader = DataLoader(DatasetSplit(dataset_train, dict_users_train[idx]), batch_size=args.local_bs, shuffle=True)
        else:
            raise NotImplementedError

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(args.device), labels.to(args.device)
            net.zero_grad()
            log_probs = net(images)
            loss = loss_func(log_probs, labels)
            _grad = torch.autograd.grad(loss, net.parameters())
            break

        for i, key in enumerate(sd_idx):
            sd_idx[key] = sd_idx[key] - args.HFMAML_alpha * _grad[i]

        net.load_state_dict(sd_idx)
        # test

        a, b = test_img_local(net, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
        tot += len(dict_users_test[idx])


        acc_test_local[idx] = a*len(dict_users_test[idx])
        loss_test_local[idx] = b*len(dict_users_test[idx])



    return sum(acc_test_local) / tot, sum(loss_test_local) / tot



# ====== TEST MODULE OF FEDME ======
def test_FEDME(net: nn.Module, args, dataset_test, dict_users_test, dataset_train, dict_users_train):
    tot = 0
    loss_func = nn.CrossEntropyLoss()
    num_idxxs = args.num_users
    acc_test_local = np.zeros(num_idxxs)
    loss_test_local = np.zeros(num_idxxs)

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
        lr=args.FEDME_lr
    )

    for idx in range(num_idxxs):


        if args.dataset == 'cifar10' or args.dataset == 'cifar100' or args.dataset == 'emnist' or args.dataset == 'mnist':
            dataloader = DataLoader(DatasetSplit(dataset_train, dict_users_train[idx]), batch_size=args.local_bs, shuffle=True)
        else:
            raise NotImplementedError

        for batch_idx, (images, labels) in enumerate(dataloader):
            images, labels = images.to(args.device), labels.to(args.device)
            net_ep = copy.deepcopy(net)  # This is fixed during a local update round
            for name, param in net.named_parameters():
                param.requires_grad = True
            for name, param in net_ep.named_parameters():
                param.requires_grad = True
            for _ in range(10):
                net.zero_grad()
                log_probs = net(images)
                loss = loss_func(log_probs, labels)
                for p_ep, p in zip(net_ep.parameters(), net.parameters()):
                    loss = loss + args.FEDME_lambda / 2 * torch.norm(p_ep - p) ** 2
                loss.backward()
                optimizer.step()


        # test

        a, b = test_img_local(net, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
        tot += len(dict_users_test[idx])


        acc_test_local[idx] = a*len(dict_users_test[idx])
        loss_test_local[idx] = b*len(dict_users_test[idx])



    return sum(acc_test_local) / tot, sum(loss_test_local) / tot