#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import argparse

def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--epochs', type=int, default=10, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=100, help="number of users: n")
    parser.add_argument('--shard_per_user', type=int, default=2, help="classes per user")
    parser.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    parser.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E; only for algorithms other than FedRep")
    parser.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    parser.add_argument('--bs', type=int, default=128, help="test batch size")
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")
    parser.add_argument('--grad_norm', action='store_true', help='use_gradnorm_avging')
    parser.add_argument('--lr_decay', type=float, default=1.0, help="learning rate decay per round")
    parser.add_argument('--local_updates', type=int, default=1000000, help="maximum number of local updates")
    parser.add_argument('--m_tr', type=int, default=500, help="maximum number of samples/user to use for training")
    parser.add_argument('--m_ft', type=int, default=500, help="maximum number of samples/user to use for fine-tuning")

    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    parser.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")
    parser.add_argument('--num_layers_keep', type=int, default=1, help='number layers to keep')
    parser.add_argument('--alg', type=str, default='fedavg', help='FL algorithm to use')
    
    # algorithm-specific hyperparameters
    parser.add_argument('--local_rep_ep', type=int, default=1, help="the number of local epochs for the representation for FedRep")
    parser.add_argument('--head_ep_per_rep_update', type=int, default=10, help="the number of local epochs for the head per representation epoch for FedRep")
    parser.add_argument('--lr_g', type=float, default=0.1, help="global learning rate for SCAFFOLD")
    parser.add_argument('--mu', type=float, default='0.1', help='FedProx parameter mu')
    parser.add_argument('--gmf', type=float, default='0', help='FedProx parameter gmf')
    parser.add_argument('--alpha_apfl', type=float, default='0.75', help='APFL parameter alpha')
    parser.add_argument('--alpha_l2gd', type=float, default='1', help='L2GD parameter alpha')
    parser.add_argument('--lambda_l2gd', type=float, default='0.5', help='L2GD parameter lambda')
    parser.add_argument('--lr_in', type=float, default='0.0001', help='PerFedAvg inner loop step size')
    parser.add_argument('--bs_frac_in', type=float, default='0.8', help='PerFedAvg fraction of batch used for inner update')
    parser.add_argument('--lam_ditto', type=float, default='1', help='Ditto parameter lambda')

    parser.add_argument('--init_clients', type=int, default=10, help="the number of clients participate initially")
    parser.add_argument('--resample', action='store_true', help="whether to resample the speed of clients")
    parser.add_argument('--double_freq', type=int, default=10, help="how often to double the participated clients")

    # other arguments
    parser.add_argument('--dataset', type=str, default='mnist', help="name of dataset")
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_classes', type=int, default=10, help="number of classes")
    parser.add_argument('--num_channels', type=int, default=3, help="number of channels of imges")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--test_freq', type=int, default=1, help='how often to test on val set')
    parser.add_argument('--load_fed', type=str, default='n', help='define pretrained federated model path')
    parser.add_argument('--results_save', type=str, default='runA', help='define fed results save folder')
    parser.add_argument('--save_every', type=int, default=50, help='how often to save models')

    parser.add_argument('--description', type=str, default='None', help='describe the setting')
    parser.add_argument('--repeat_id', type=int, default=0, help='the repeat id of the run')
    parser.add_argument('--FT_epoch', type=int, default=10, help='the number of epoch for fine tuning')
    parser.add_argument('--hyper_setting', type=str, default='iid-hyper', choices=['iid-hyper', 'noniid-hyper'])
    parser.add_argument('--hyper_high', type=float, default='10')
    parser.add_argument('--hyper_low', type=float, default='1')
    args = parser.parse_args()
    return args
