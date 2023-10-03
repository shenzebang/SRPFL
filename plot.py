import matplotlib.pyplot as plt
import matplotlib as mpl

import csv
import numpy as np

import os

mpl.rcParams['savefig.pad_inches'] = 0

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 25

def find_first(xs, target):
    for i in range(xs.shape[0]):
        if xs[i] > target:
            return i


# settings = (
#     (2, 100, "cifar10"),
#     (2, 500, "cifar10"),
#     (5, 100, "cifar10"),
#     (5, 100, "cifar100"),
#     (20, 100, "cifar100"),
# )

settings = (
    # (5, 100, "cifar10"),
    # (2, 100, "cifar10"),
    # (10, 100, "cifar10", 1.0),
    # (10, 100, "emnist",  1.0),
    # (3,  100, "femnist", 1.0),
    (5, 100, "cifar10", 1.0),
)

# settings = (
#     # (5, 500),
#     # (2, 500),
#     (2, 94, "emnist"),
#     (5, 94, "emnist"),
#     (2, 470, "emnist"),
#     (5, 470, "emnist")
# )


# settings = (
#     (2, 100, "mnist"),
# )
# datasets = ["cifar100"]
# communication_times = (0, 1000, 10000, 100000)
communication_times = (0, 1, 10, 100)
# communication_times = (0, )
# shards = [2, 5]
# methods = ["ffgd", "fed-avg", "scaffold"]
# descriptions = ["fedrep-double-10-10-s1", "fedrep-full-s1", "fedavg-double-10-10-s1", "fedavg-full-s1"]
# descriptions = ["fedrep-full-s1", "fedavg-double-10-10-s1", "fedavg-full-s1"]
# descriptions = ["fedavg-double-10-10-s1-FT", "fedrep-double-10-10-s1-FT", "fedrep-full-s1-FT", "fedavg-full-s1-FT",
#                 "fedavg-double-10-10-s1-global", "fedavg-full-s1-global"]

# methods = ["flanp"]
# methods = ["FedRep"]
# methods_label = {"flanp" : "FLANP", "FedRep": "FedRep"}
# colors = ['red', 'darkred', 'blue', 'darkblue', 'green', 'darkgreen', 'orange', 'darkorange', 'pink']

colors = {
    "FEDREP-s1-FT":         'darkred',
    "FEDREP-s1-flanp-FT":   'red',
    "FEDAVG-s1-FT":         'darkorange',
    "FEDAVG-s1-flanp-FT":   'orange',
    "HFMAML-s1-global":     'pink',
    "FEDAVG-s1-global":     'darkgreen',
    "FEDAVG-s1-flanp-global":   'green',
    "LG-s1-flanp-FT":       'blue',
    "LG-s1-FT":             'darkblue',
}
first = True
# first = False
# partial = True
partial = False
hyper = 'iid-hyper'
# mode = 'fast-and-slow'
mode = ''
# hyper = 'noniid-hyper'
s = 's1' if hyper == 'iid-hyper' else 's2'
# LABELS = {
#     "FEDREP-s1-FT":         'FedRep (Collins et al., 2021)',
#     "FEDREP-s1-flanp-FT":   "FedRep-SRPFL (Our method)",
#     "FEDAVG-s1-FT":         'FedAvg-ft (Yu et al., 2020)',
#     "FEDAVG-s1-flanp-FT":   'FLANP-ft (Reisizadeh et al., 2022)',
#     "HFMAML-s1-global":     'HFMAML (Fallah et al., 2020)',
#     "FEDAVG-s1-global":     'FedAvg (McMahan et al., 2017)',
#     "FEDAVG-s1-flanp-global":   'FLANP (Reisizadeh et al., 2022)',
#     "LG-s1-flanp-FT":       'LG-SRPFL',
#     "LG-s1-FT":             'LG (Liang et al., 2020)',
# }
LABELS = {
    "FEDREP-s1-FT":         'FedRep',
    "FEDREP-s1-flanp-FT":   "FedRep-SRPFL",
    "FEDAVG-s1-FT":         'FedAvg-FT',
    "FEDAVG-s1-flanp-FT":   'FLANP-FT',
    "HFMAML-s1-global":     'HFMAML',
    "FEDAVG-s1-global":     'FedAvg',
    "FEDAVG-s1-flanp-global":   'FLANP',
    "LG-s1-flanp-FT":       'LG-SRPFL',
    "LG-s1-FT":             'LG-FedAvg',
}
for communication_time in communication_times:
    for shard, N, dataset, participation_rate in settings:
        # descriptions = ["FEDREP", "AFA-CD", "FEDASYNC", "FEDREP-SRPFL"]
        descriptions = ["FEDREP-s1-flanp-FT", "FEDREP-s1-FT",
                        "LG-s1-flanp-FT", "LG-s1-FT",
                        "FEDAVG-s1-flanp-global", "FEDAVG-s1-flanp-FT",
                        "FEDAVG-s1-global", "FEDAVG-s1-FT",
                        "HFMAML-s1-global",
        ]
        color_id = 0
        min_comm = 10000000
        for description in descriptions:
            # print(method)
            color = colors[description]
            color_id += 1
            comms = []
            accus = []
            min_len = 10000000
            for j in range(1, 5):
                # file_name = f'save/sent140_{description}_{mode}_.csv'
                file_name = f'save/{dataset}-{shard}-{N}-{participation_rate}/{description}-{j}-iid-hyper.csv'
                with open(file_name, 'r') as csvfile:
                    plots = csv.reader(csvfile, delimiter=',')
                    i = 0
                    comm = []
                    accu = []
                    for idx, row in enumerate(plots):
                        if i == 0: # ignore the title (the first row)
                            i += 1
                            continue
                        comm.append(float(row[0]) + idx * communication_time)
                        accu.append(float(row[1]))
                    min_len = min(min_len, len(accu))
                    min_comm = min(min_comm, max(comm))
                accus.append(accu)
            accus = [accu[:min_len] for accu in accus]
            comm = comm[:min_len]
            # FT_acc = np.mean(np.asarray(accus), axis=0)[-1]
            # print(f"{description} FT acc {FT_acc}")
            # accus = [accu[: -1] for accu in accus]
            # comm = comm[: -1]
            accus_array = np.asarray(accus)
            # accu_mean = np.mean(accus_array)
            accu_mean = np.mean(accus_array, axis=0)
            print(f"{description} max acc {np.max(accu_mean)}")
            accu_std = np.std(accus_array, axis=0)
            linestyle = 'solid' if 'flanp' in description else 'dashed'
            # if description == "FEDREP-SRPFL":
            #     label = "FedRep-SRPFL (Our method)"
            # elif description == "FEDREP":
            #     label = "FedRep (Collins et al., 2021)"
            # elif description == "AFA-CD":
            #     label = "AFA-CD (Yang et al., 2022)"
            # elif description == "FEDASYNC":
            #     label = "FedAsync (Xie et al., 2020)"

            label = LABELS[description]

            plt.plot(comm, accu_mean, label=label, color=color, linewidth=5.0, linestyle=linestyle)

            # plt.fill_between(comm, accu_mean - accu_std, accu_mean + accu_std, facecolor=color, alpha=0.15)
            # print(comm[find_first(accu_mean, 0.495)])
            # plt.plot(comm[1:], accu[1:], label=method)

            # ax = plt.gca()



        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
        # plt.ticklabel_format(axis='x', style='sci',)
        plt.xticks(np.arange(0, int(min_comm)+1, int(min_comm/4)).astype(int), size=BIGGER_SIZE)
        # plt.yticks(np.arange(start=0.6, stop=0.83, step=0.05), size=BIGGER_SIZE)
        # if participation_rate == 1.0:
        #     plt.xlabel(f'Homogeneous (C.T. = {communication_time})', size=BIGGER_SIZE)
        # else:
        plt.xlabel(f'Total Time (C.T. = {communication_time})', size=BIGGER_SIZE)
        plt.xlim(0, min_comm)
        plt.ylabel('Testing Accuracy', size=BIGGER_SIZE)
        if dataset == 'cifar10':
            _dataset = 'CIFAR10'
        elif dataset == 'cifar100':
            _dataset = 'CIFAR100'
        elif dataset == 'emnist':
            _dataset = 'EMNIST'
        elif dataset == 'sent140' or dataset == 'sent140_homo':
            _dataset = 'SENT140'
        elif dataset == 'femnist':
            _dataset = 'FEMNIST'
        else:
            raise NotImplementedError
        plt.title(f'{_dataset}, M={N}, Shard={shard}', size=BIGGER_SIZE)
        # plt.xticks()
        plt.yticks(size=BIGGER_SIZE)
        if first:
            plt.legend()
            plt.legend(fontsize='x-large')
            first = False
        # plt.yticks(np.arange(0, 11, 400))
        # plt.xticks(np.arange(0, 2001, 400))

        save_directory = './plot/'
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        if partial:
            plt.savefig(f'plot/test_acc_time_{dataset}_{N}_shard{shard}-{communication_time}-partial-participation.pdf', bbox_inches='tight')
        else:
            plt.savefig(f'plot/test_acc_time_{dataset}_{N}_shard{shard}-{communication_time}_{mode}.pdf',
                        bbox_inches='tight')
        plt.show()


# plt.errorbar(ffgd_comm[1:], ffgd_accu[1:], yerr=ffgd_yerr[1:], label='ffgd-0.3')
# plt.plot(sgd_time[1:], sgd_loss[1:], label='Adam')
# ax.set_yscale('log')
# plt.yscale('log')


