

## Usage

To replicate the results presented in our submission, please use the "*.sh" files under the experiments folder.

## Datasets

The CIFAR10, CIFAR100 AND MNIST datasets are downloaded automatically by the torchvision package. 
FEMNIST is provided by the LEAF repository, which should be downloaded from https://github.com/TalwalkarLab/leaf/ and placed in `FedRep/`. 
Then the raw FEMNIST dataset can be downloaded by following the instructions in LEAF. 
In order to generate the versions of these datasets that we use the paper, we use the following commands from within `FedRep/leaf-master/data/femnist/`:


FEMNIST: `./preprocess.sh -s niid --sf 0.5 -k 50 -tf 0.8 -t sample`

For FEMNIST, we re-sample and re-partition the data to increase its heterogeneity. In order to do so, first navigate to `FedRep/`, then execute 

`mv my_sample.py leaf/data/femnist/data/`

`cd leaf/data/femnist/data/`

`python my_sample.py`