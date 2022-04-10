# Modified from: https://github.com/pliang279/LG-FedAvg/blob/master/models/Update.py
# credit goes to: Paul Pu Liang

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy
import FedProx
import torch.nn.functional as F

from models.test import test_img_local
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y 

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]),(1,28,28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdateMAML(object):

    def __init__(self, args, dataset=None, idxs=None, optim=None,indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.optim = optim
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net, c_list={}, idx=-1, lr=0.1,lr_in=0.0001, c=False):
        net.train()
        # train and update
        lr_in = lr*0.001
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )
        
        local_eps = self.args.local_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(2)
        for iter in range(local_eps):
            batch_loss = []
            if num_updates == self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)

                    split = self.args.local_bs 
                    sup_x, sup_y = data.to(self.args.device), targets.to(self.args.device)
                    targ_x, targ_y = data.to(self.args.device), targets.to(self.args.device)

                    param_dict = dict()
                    for name, param in net.named_parameters():
                        if param.requires_grad:
                            if "norm_layer" not in name:
                                param_dict[name] = param.to(device=self.args.device)
                    names_weights_copy = param_dict

                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    log_probs_sup = net(sup_x, hidden_train)
                    loss_sup = self.loss_func(log_probs_sup,sup_y)
                    grads = torch.autograd.grad(loss_sup, names_weights_copy.values(),
                                                    create_graph=True, allow_unused=True)
                    names_grads_copy = dict(zip(names_weights_copy.keys(), grads))

                    for key, grad in names_grads_copy.items():
                        if grad is None:
                            print('Grads not found for inner loop parameter', key)
                        names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                    for key in names_grads_copy.keys():
                        names_weights_copy[key] = names_weights_copy[key]- lr_in * names_grads_copy[key]

                    log_probs_targ = net(targ_x)
                    loss_targ = self.loss_func(log_probs_targ,targ_y)
                    loss_targ.backward()
                    optimizer.step()
                        
                    del log_probs_targ.grad
                    del loss_targ.grad
                    del loss_sup.grad
                    del log_probs_sup.grad
                    optimizer.zero_grad()
                    net.zero_grad()

                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    split = int(8* images.size()[0]/10)
                    sup_x, sup_y = images[:split].to(self.args.device), labels[:split].to(self.args.device)
                    targ_x, targ_y = images[split:].to(self.args.device), labels[split:].to(self.args.device)

                    param_dict = dict()
                    for name, param in net.named_parameters():
                        if param.requires_grad:
                            if "norm_layer" not in name:
                                param_dict[name] = param.to(device=self.args.device)
                    names_weights_copy = param_dict

                    net.zero_grad()
                    log_probs_sup = net(sup_x)
                    loss_sup = self.loss_func(log_probs_sup,sup_y)
                    if loss_sup != loss_sup:
                        continue
                    grads = torch.autograd.grad(loss_sup, names_weights_copy.values(),
                                                    create_graph=True, allow_unused=True)
                    names_grads_copy = dict(zip(names_weights_copy.keys(), grads))
                        
                    for key, grad in names_grads_copy.items():
                        if grad is None:
                            print('Grads not found for inner loop parameter', key)
                        names_grads_copy[key] = names_grads_copy[key].sum(dim=0)
                    for key in names_grads_copy.keys():
                        names_weights_copy[key] = names_weights_copy[key]- lr_in * names_grads_copy[key]
                        
                    loss_sup.backward(retain_graph=True)
                    log_probs_targ = net(targ_x)
                    loss_targ = self.loss_func(log_probs_targ,targ_y)
                    loss_targ.backward()
                    optimizer.step()
                    del log_probs_targ.grad
                    del loss_targ.grad
                    del loss_sup.grad
                    del log_probs_sup.grad
                    optimizer.zero_grad()
                    net.zero_grad()
 
                batch_loss.append(loss_sup.item())
                num_updates += 1
                if num_updates == self.args.local_updates:
                    break
                batch_loss.append(loss_sup.item())
                
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd#, num_updates

class LocalUpdateFedPD:
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, lambdai):
        net_local = copy.deepcopy(net)
        net_local.requires_grad_(True)
        net.requires_grad_(False)
        bias_p = []
        weight_p = []
        for name, p in net_local.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=self.args.lr, momentum=self.args.momentum
        )

        local_eps = self.args.local_ep

        total_samples = 0
        total_loss = 0

        state_dict_local = net_local.state_dict()
        state_dict_global = net.state_dict()

        for iter in range(local_eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net_local.zero_grad()
                log_probs = net_local(images)
                loss = self.loss_func(log_probs, labels)

                total_loss += loss.item()
                total_samples += len(labels)

                # add the linear penalty
                linear_penalty = 0.
                for key in state_dict_local.keys():
                    linear_penalty += torch.sum(state_dict_local[key] * lambdai[key])
                loss += linear_penalty

                quad_penalty = 0.
                for key in state_dict_local.keys():
                    quad_penalty += F.mse_loss(state_dict_local[key], state_dict_global[key], reduction='sum')

                loss += quad_penalty / 2. / self.args.FedPD_eta


                loss.backward()
                optimizer.step()

        # update the dual variable
        with torch.autograd.no_grad():
            for key in state_dict_local.keys():
                lambdai[key] += (state_dict_local[key] - state_dict_global[key])/self.args.FedPD_eta

            for key in state_dict_local.keys():
                state_dict_local[key] += self.args.FedPD_eta * lambdai[key]

        return state_dict_local, lambdai, total_loss/total_samples




class LocalUpdateScaffold(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net, c_list={}, idx=-1, lr=0.1, c=False):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )
        
        local_eps = self.args.local_ep

        epoch_loss=[]
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            batch_loss = []
            if num_updates == self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()

                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss_fi = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    w = net.state_dict()
                    local_par_list = None
                    dif = None
                    for param in net.parameters():
                        if not isinstance(local_par_list, torch.Tensor):
                            local_par_list = param.reshape(-1)
                        else:
                            local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                    for k in c_list[idx].keys():
                        if not isinstance(dif, torch.Tensor):
                            dif = (-c_list[idx][k] +c_list[-1][k]).reshape(-1)
                        else:
                            dif = torch.cat((dif, (-c_list[idx][k]+c_list[-1][k]).reshape(-1)),0)
                    loss_algo = torch.sum(local_par_list * dif)
                    loss = loss_fi + loss_algo
                    
                    loss.backward()
                    optimizer.step()

                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)

                    log_probs = net(images)
                    loss_fi = self.loss_func(log_probs, labels)
                    w = net.state_dict()
                    local_par_list = None
                    dif = None
                    for param in net.parameters():
                        if not isinstance(local_par_list, torch.Tensor):
                            local_par_list = param.reshape(-1)
                        else:
                            local_par_list = torch.cat((local_par_list, param.reshape(-1)), 0)

                    for k in c_list[idx].keys():
                        if not isinstance(dif, torch.Tensor):
                            dif = (-c_list[idx][k] +c_list[-1][k]).reshape(-1)
                        else:
                            dif = torch.cat((dif, (-c_list[idx][k]+c_list[-1][k]).reshape(-1)),0)
                    loss_algo = torch.sum(local_par_list * dif)
                    loss = loss_fi + loss_algo
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=net.parameters(), max_norm=10)
                    optimizer.step()

                num_updates += 1
                if num_updates == self.args.local_updates:
                    break
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd, num_updates

class LocalUpdateAPFL(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net,ind=None,w_local=None, idx=-1, lr=0.1):
        net.train()
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )
        
        # train and update
        local_eps = self.args.local_ep
        args = self.args
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            batch_loss = []
            if num_updates >= self.args.local_updates:
                break
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if  'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    w_loc_new = {}
                    w_glob = copy.deepcopy(net.state_dict())
                    for k in net.state_dict().keys():
                        w_loc_new[k] = self.args.alpha_apfl*w_local[k] + self.args.alpha_apfl*w_glob[k]

                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    optimizer.zero_grad()
                    loss.backward()
                        
                    optimizer.step()
                    optimizer.zero_grad()
                    wt = copy.deepcopy(net.state_dict())
                    net.zero_grad()

                    del hidden_train
                    hidden_train = net.init_hidden(self.args.local_bs)

                    net.load_state_dict(w_loc_new)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.args.alpha_apfl*self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                    w_local_bar = net.state_dict()
                    for k in w_local_bar.keys():
                        w_local[k] = w_local_bar[k] - w_loc_new[k] + w_local[k]

                    net.load_state_dict(wt)
                    optimizer.zero_grad()
                    del wt
                    del w_loc_new
                    del w_glob
                    del w_local_bar
                    
                else:
                        
                    w_loc_new = {} 
                    w_glob = copy.deepcopy(net.state_dict())
                    for k in net.state_dict().keys():
                        w_loc_new[k] = self.args.alpha_apfl*w_local[k] + self.args.alpha_apfl*w_glob[k]

                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                        
                    optimizer.step()
                    wt = copy.deepcopy(net.state_dict())

                    net.load_state_dict(w_loc_new)
                    log_probs = net(images)
                    loss = self.args.alpha_apfl*self.loss_func(log_probs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    w_local_bar = net.state_dict()
                    for k in w_local_bar.keys():
                        w_local[k] = w_local_bar[k] - w_loc_new[k] + w_local[k]

                    net.load_state_dict(wt)
                    optimizer.zero_grad()
                    del wt
                    del w_loc_new
                    del w_glob
                    del w_local_bar

                num_updates += 1
                if num_updates >= self.args.local_updates:
                    break

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(),w_local, sum(epoch_loss) / len(epoch_loss), self.indd

class LocalUpdateDitto(object):

    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None

    def train(self, net,ind=None, w_ditto=None, lam=0, idx=-1, lr=0.1, last=False):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=0.5
        )

        local_eps = self.args.local_ep
        args = self.args 
        epoch_loss=[]
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done=False
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    w_0 = copy.deepcopy(net.state_dict())
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()

                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train) 
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()

                    if w_ditto is not None:
                        w_net = copy.deepcopy(net.state_dict())
                        for key in w_net.keys():
                            w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key])

                        net.load_state_dict(w_net)
                        optimizer.zero_grad()
                else:
                    w_0 = copy.deepcopy(net.state_dict())
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    if w_ditto is not None:
                        w_net = copy.deepcopy(net.state_dict())
                        for key in w_net.keys():
                            w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key])
                        net.load_state_dict(w_net)
                        optimizer.zero_grad()
                
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates >= self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if done:
                break
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd


class LocalUpdateNova(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs

    def train(self, net_global, local_ep, lr=0.1):
        bias_p = []
        weight_p = []
        net_local = copy.deepcopy(net_global)
        for name, p in net_local.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=self.args.momentum
        )

        local_eps = local_ep

        epoch_loss = []
        num_updates = 0

        for name, param in net_local.named_parameters():
            param.requires_grad = True

        total_samples = 0
        total_loss = 0
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net_local.zero_grad()
                log_probs = net_local(images)
                loss = self.loss_func(log_probs, labels)
                total_samples += len(labels)
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                num_updates += 1

        sdl = net_local.state_dict()
        sdg = net_global.state_dict()
        for key in sdl.keys():
            sdl[key] -= sdg[key]

        return sdl, total_loss / total_samples, self.indd


class LocalUpdateMTL(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        else:
            self.indd = indd

    def train(self, net, lr=0.1, omega=None, W_glob=None, idx=None, w_glob_keys=None):
        net.train()
        # train and update
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name or name in w_glob_keys:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )

        epoch_loss = []
        local_eps = self.args.local_ep
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break

                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(
                        self.args.device)
                    net.zero_grad()

                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    W = W_glob.clone()
                    W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                    W_local = torch.cat(W_local)
                    W[:, idx] = W_local

                    loss_regularizer = 0
                    loss_regularizer += W.norm() ** 2

                    k = 4000
                    for i in range(W.shape[0] // k):
                        x = W[i * k:(i + 1) * k, :]
                        loss_regularizer += x.mm(omega).mm(x.T).trace()
                    f = (int)(math.log10(W.shape[0]) + 1) + 1
                    loss_regularizer *= 10 ** (-f)

                    loss = loss + loss_regularizer
                    loss.backward()
                    optimizer.step()

                else:

                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    W = W_glob.clone().to(self.args.device)
                    W_local = [net.state_dict(keep_vars=True)[key].flatten() for key in w_glob_keys]
                    W_local = torch.cat(W_local)
                    W[:, idx] = W_local

                    loss_regularizer = 0
                    loss_regularizer += W.norm() ** 2

                    k = 4000
                    for i in range(W.shape[0] // k):
                        x = W[i * k:(i + 1) * k, :]
                        loss_regularizer += x.mm(omega).mm(x.T).trace()
                    f = (int)(math.log10(W.shape[0]) + 1) + 1
                    loss_regularizer *= 10 ** (-f)

                    loss = loss + loss_regularizer
                    loss.backward()
                    optimizer.step()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd

# Generic local update class, implements local updates for FedRep, FedPer, LG-FedAvg, FedAvg, FedProx
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset: 
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])),name=self.args.dataset), batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
         
        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd=None        
        
        self.dataset=dataset
        self.idxs=idxs

    def train(self, net, representation_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
        [     
            {'params': weight_p, 'weight_decay':0.0001},
            {'params': bias_p, 'weight_decay':0}
        ],
        lr=lr, momentum=self.args.momentum
        )
        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                             lr=lr,
                             gmf=self.args.gmf,
                             mu=self.args.mu,
                             ratio=1/self.args.num_users,
                             momentum=0.5,
                             nesterov = False,
                             weight_decay = 1e-4)

        is_alternating_update = self.args.alg == "fedrep"

        if is_alternating_update:
            local_eps = self.args.local_rep_ep * (self.args.head_ep_per_rep_update+1)
        else:
            local_eps = self.args.local_ep

        # if last:
        #     if self.args.alg =='fedavg' or self.args.alg == 'prox':
        #         local_eps= 10
        #         net_keys = [*net.state_dict().keys()]
        #         if 'cifar' in self.args.dataset:
        #             w_glob_keys = [net.weight_keys[i] for i in [0,1,3,4]]
        #         elif 'sent140' in self.args.dataset:
        #             w_glob_keys = [net_keys[i] for i in [0,1,2,3,4,5]]
        #         elif 'mnist' in self.args.dataset:
        #             w_glob_keys = [net.weight_keys[i] for i in [0,1,2]]
        #     elif 'maml' in self.args.alg:
        #         local_eps = 5
        #         w_glob_keys = []
        #     else:
        #         local_eps =  10

        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False
            flag_update_head = iter % (1+self.args.head_ep_per_rep_update) != self.args.head_ep_per_rep_update
            # for FedRep, first do local epochs for the head
            if flag_update_head and is_alternating_update:
                for name, param in net.named_parameters():
                    if name in representation_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
            
            # then do local epochs for the representation
            elif not flag_update_head and is_alternating_update:
                for name, param in net.named_parameters():
                    if name in representation_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            # all other methods update all parameters simultaneously
            elif not is_alternating_update:
                for name, param in net.named_parameters():
                     param.requires_grad = True 
       
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels,self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])
                    loss.backward()
                    optimizer.step()
                else:
                    images, labels = images.to(self.args.device), labels.to(self.args.device)
                    net.zero_grad()
                    log_probs = net(images)
                    loss = self.loss_func(log_probs, labels)
                    loss.backward()
                    optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            if done:
                break
            
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd

# local update class for FEDREP
class LocalUpdateFEDAVG(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.dataset = dataset
        self.idxs = idxs

    def train(self, net):
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
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

# local update class for FEDREP
class LocalUpdateFEDREP(object):
    def __init__(self, args, dataset, idxs, representation_keys):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.representation_keys = representation_keys
        self.dataset = dataset
        self.idxs = idxs

    def train(self, net):
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
            lr=self.args.lr, momentum=self.args.momentum
        )

        local_eps = self.args.FEDREP_local_rep_ep * (self.args.FEDREP_head_ep_per_rep_update + 1)

        epoch_loss = []
        num_updates = 0

        for iter in range(local_eps):
            done = False
            flag_update_head = iter % (1 + self.args.FEDREP_head_ep_per_rep_update) != self.args.FEDREP_head_ep_per_rep_update
            # first do local epochs for the head
            if flag_update_head:
                for name, param in net.named_parameters():
                    if name in self.representation_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif not flag_update_head :
                for name, param in net.named_parameters():
                    if name in self.representation_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                num_updates += 1
                batch_loss.append(loss.item())
                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

# local update class for HF-MAML
class LocalUpdateHFMAML:
    def __init__(self, args, dataset, idxs):
        # define the dataloaders for training
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.HFMAML_local_bs, shuffle=True)
        self.idxs = idxs

    def train(self, net: nn.Module):
        local_eps = self.args.local_ep

        for ep in range(local_eps):
            for images, labels in self.ldr_train:
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                # named_param_temp is named_param after a gradient step
                sd = net.state_dict()
                sd_temp = copy.copy(sd) # only copy the key and the references to the tensors
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                grad_at_named_param = torch.autograd.grad(loss, net.parameters())
                for idx, key in enumerate(sd_temp):
                    sd_temp[key] = sd_temp[key] - self.args.HFMAML_alpha * grad_at_named_param[idx]
                del grad_at_named_param # remove to save memory

                # compute the gradient at named_param_temp
                net.load_state_dict(sd_temp)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                grad_at_named_param_temp = torch.autograd.grad(loss, net.parameters())


                # compute the Hessian-vector product via gradient difference

                    # named_param_pg is param + delta * grad_at_named_param_temp
                sd_pg = copy.copy(sd)
                for idx, key in enumerate(sd_pg):
                    sd_pg[key] = sd_pg[key] + self.args.HFMAML_delta * grad_at_named_param_temp[idx]
                net.load_state_dict(sd_pg)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                grad_pg = torch.autograd.grad(loss, net.parameters())
                del sd_pg

                    # named_param_mg is param - delta * grad_at_named_param_temp
                sd_mg = copy.copy(sd)
                for idx, key in enumerate(sd_mg):
                    sd_mg[key] = sd_mg[key] - self.args.HFMAML_delta * grad_at_named_param_temp[idx]
                net.load_state_dict(sd_mg)
                net.zero_grad()
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                grad_mg = torch.autograd.grad(loss, net.parameters())
                del sd_mg

                    # hvp: hessian vector product
                hvp = copy.copy(sd)
                for idx, key in enumerate(hvp):
                    hvp[key] = (grad_pg[idx] - grad_mg[idx]) / 2. / self.args.HFMAML_delta
                del grad_pg, grad_mg

                # update the variable
                for idx, key in enumerate(sd):
                    sd[key] = sd[key] - self.args.HFMAML_beta * (
                        grad_at_named_param_temp[idx] - self.args.HFMAML_alpha * hvp[key]
                    )

        return sd, 0.




