# FEDREP-SRPFL
python FEDREP-uniform-ray.py --flanp --dataset cifar10 --model cnn --num_classes 10 --epochs 200 --double_freq 15 --init_clients 10 --num_users 100 --gpu 0 --seed 1 --shard_per_user 2 --test_freq 10 --frac 1 --repeat_id 1 --description ICML_Rebuttal_FEDREP_SRPFL --hyper_setting iid-hyper --FEDREP_local_rep_ep 1 --FEDREP_head_ep_per_rep_update 10 --FT_epoch 10&
# FEDREP
python FEDREP-uniform-ray.py --dataset cifar10 --model cnn --num_classes 10 --epochs 200 --double_freq 10 --init_clients 100 --num_users 100 --gpu 1 --seed 1 --shard_per_user 2 --test_freq 10 --frac 1 --repeat_id 1 --description ICML_Rebuttal_FEDREP --hyper_setting iid-hyper --FEDREP_local_rep_ep 1 --FEDREP_head_ep_per_rep_update 10 --FT_epoch 10&
# AFA-CD
python AFA-CD.py --dataset cifar10 --model cnn --num_classes 10 --maximum_system_time 20000 --num_users 100 --gpu 2 --seed 1 --shard_per_user 2 --test_freq 1000 --frac 1 --repeat_id 1 --description ICML_Rebuttal_AFA_CD --hyper_setting iid-hyper --FEDAVG_local_ep 1 --FT_epoch 10&
# FEDASYNC
python FEDASYNC.py --dataset cifar10 --model cnn --num_classes 10 --maximum_system_time 20000 --num_users 100 --gpu 3 --seed 1 --shard_per_user 2 --test_freq 1000 --frac 1 --repeat_id 1 --description ICML_Rebuttal_FEDASYNC --hyper_setting iid-hyper --FEDAVG_local_ep 1 --FT_epoch 10&