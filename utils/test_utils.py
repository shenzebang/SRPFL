import numpy as np
import copy
from torch.utils.data import DataLoader
from models.Update import DatasetSplit
import torch
import torch.nn as nn
from models.test import test_img_local


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
        elif args.dataset == "femnist":
            dataset_train_idx = dataset_train[list(dataset_train.keys())[idx]]
            dataloader = DataLoader(DatasetSplit(dataset_train_idx, np.ones(len(dataset_train_idx['x'])),name=args.dataset), batch_size=args.local_bs, shuffle=True)
        else:
            raise NotImplementedError

        fine_tune(net_local, args, dataloader, representation_keys)

        # test

        net_local.eval()

        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            a, b =  test_img_local(net_local, dataset_test, args,idx=dict_users_test[idx],indd=indd, user_idx=idx)
            tot += len(dataset_test[dict_users_test[idx]]['x'])
        else:
            a, b = test_img_local(net_local, dataset_test, args, user_idx=idx, idxs=dict_users_test[idx])
            tot += len(dict_users_test[idx])

        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            acc_test_local[idx] = a*len(dataset_test[dict_users_test[idx]]['x'])
            loss_test_local[idx] = b*len(dataset_test[dict_users_test[idx]]['x'])
        else:
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