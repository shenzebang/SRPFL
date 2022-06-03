# CIFAR 10
for i in {1..4}
do
    python FEDREP-ray.py\
        --dataset cifar10 --model cnn --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 5 --test_freq 10 --frac .2 --repeat_id $i --description s1 --hyper_setting noniid-hyper\
        --FEDREP_local_rep_ep 1 --FEDREP_head_ep_per_rep_update 10 --FT_epoch 10&
done
for i in {1..4}
do
    python FEDREP-ray.py\
        --dataset cifar10 --model cnn --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 5 --test_freq 10 --frac .2 --repeat_id $i --description s1-flanp --hyper_setting noniid-hyper\
        --FEDREP_local_rep_ep 1 --FEDREP_head_ep_per_rep_update 10 --FT_epoch 10 --flanp&
done
for i in {1..4}
do
    python FEDAVG-ray.py\
        --dataset cifar10 --model cnn --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 5 --test_freq 10 --frac .2 --repeat_id $i --description s1 --hyper_setting noniid-hyper\
        --FEDAVG_local_ep 1 --FT_epoch 10&
done
for i in {1..4}
do
    python FEDAVG-ray.py\
        --dataset cifar10 --model cnn --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 5 --test_freq 10 --frac .2 --repeat_id $i --description s1-flanp --hyper_setting noniid-hyper\
        --FEDAVG_local_ep 1 --FT_epoch 10 --flanp&
done
for i in {1..4}
do
    python LG-ray.py\
        --dataset cifar10 --model cnn --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 5 --test_freq 10 --frac .2 --repeat_id $i --description s1 --hyper_setting noniid-hyper\
        --LG_local_ep 1 --FT_epoch 10&
done
for i in {1..4}
do
    python LG-ray.py\
        --dataset cifar10 --model cnn --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 5 --test_freq 10 --frac .2 --repeat_id $i --description s1-flanp --hyper_setting noniid-hyper\
        --LG_local_ep 1 --FT_epoch 10 --flanp&
done
for i in {1..4}
do
    python HFMAML-ray.py\
        --dataset cifar10 --model cnn --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 5 --test_freq 10 --frac .2 --repeat_id $i --description s1 --hyper_setting noniid-hyper\
        --HFMAML_alpha 1e-2 --HFMAML_beta 1e-1 --HFMAML_delta 1e-3 --HFMAML_local_bs 10 --HFMAML_local_ep 1 --FT_epoch 10&
done