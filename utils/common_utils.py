import numpy as np
import pandas as pd
import os
from typing import Dict

from utils.train_utils import get_data, read_data

def save_results(args, algo, results: Dict):
    save_dir = f"./save/{args.dataset}-{args.shard_per_user}-{args.num_users}-{args.frac}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    running_time_record = results["running_time_record"]
    times = np.array(running_time_record)

    if "FT_accs" in results:
        FT_accs = results["FT_accs"]
        FT_save_file = save_dir + f"/{algo}-{args.description}-FT-{args.repeat_id}-{args.hyper_setting}.csv"
        FT_accs = np.array(FT_accs)
        FT_accs = pd.DataFrame(np.stack([times, FT_accs], axis=1), columns=['times', 'accs'])
        FT_accs.to_csv(FT_save_file, index=False)

    if "global_accs" in results:
        global_accs = results["global_accs"]
        global_save_file = save_dir + f"/{algo}-{args.description}-global-{args.repeat_id}-{args.hyper_setting}.csv"
        global_accs = np.array(global_accs)
        global_accs = pd.DataFrame(np.stack([times, global_accs], axis=1), columns=['times', 'accs'])
        global_accs.to_csv(global_save_file, index=False)


def load_dataset(args):
    if 'cifar' in args.dataset or args.dataset == 'mnist' or args.dataset == 'emnist':
        # lens = np.ones(args.num_users)
        n_local_datapoints = []
        dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
        for idx in dict_users_train.keys():
            np.random.shuffle(dict_users_train[idx])
            n_local_datapoints.append(len(dict_users_train[idx]))
        print(n_local_datapoints)
    else:
        if 'femnist' in args.dataset:
            train_path = f'data/femnist/mytrain'
            test_path = f'data/femnist/mytest'
        else:
            raise NotImplementedError

        clients, groups, dataset_train, dataset_test = read_data(train_path, test_path)
        n_local_datapoints = []
        for iii, c in enumerate(clients):
            n_local_datapoints.append(len(dataset_train[c]['x']))
        dict_users_train = list(dataset_train.keys())
        dict_users_test = list(dataset_test.keys())
        print(n_local_datapoints)
        print(clients)
        for c in dataset_train.keys():
            dataset_train[c]['y'] = list(np.asarray(dataset_train[c]['y']).astype('int64'))
            dataset_test[c]['y'] = list(np.asarray(dataset_test[c]['y']).astype('int64'))

    return dataset_train, dataset_test, dict_users_train, dict_users_test, n_local_datapoints

def resample_simulated_running_time(args, simulated_running_time):
    if args.hyper_setting == "iid-hyper":
        if args.resample:
            # regenerate samples from expotential distribution
            simulated_running_time = np.random.exponential(1, args.num_users)
    elif args.hyper_setting == "noniid-hyper":
        if args.resample:
            exp_hypers = np.random.uniform(low=args.hyper_low, high=args.hyper_high, size=(args.num_users,))
        # the exp_hypers should be input.
        simulated_running_time = np.squeeze(np.array([np.random.exponential(hyper, 1) for hyper in exp_hypers]))
    else:
        raise NotImplementedError

    return simulated_running_time

def get_simulated_running_time(args):
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

    return simulated_running_time
