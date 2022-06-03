# CIFAR 10
for i in {1..4}
do
    python FEDREP-ray.py\
        --dataset femnist --model mlp --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 3 --test_freq 10 --frac 1 --repeat_id $i --description s1 --hyper_setting noniid-hyper\
        --FEDREP_local_rep_ep 5 --FEDREP_head_ep_per_rep_update 10 --FT_epoch 10\
        --leaf_path /home/lab304/zebang/dataset&
done
for i in {1..4}
do
    python FEDREP-ray.py\
        --dataset femnist --model mlp --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 3 --test_freq 10 --frac 1 --repeat_id $i --description s1-flanp --hyper_setting noniid-hyper\
        --FEDREP_local_rep_ep 5 --FEDREP_head_ep_per_rep_update 10 --FT_epoch 10 --flanp\
        --leaf_path /home/lab304/zebang/dataset&
done
for i in {1..4}
do
    python FEDAVG-ray.py\
        --dataset femnist --model mlp --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 3 --test_freq 10 --frac 1 --repeat_id $i --description s1 --hyper_setting noniid-hyper\
        --FEDAVG_local_ep 5 --FT_epoch 10 --leaf_path /home/lab304/zebang/dataset&
done
for i in {1..4}
do
    python FEDAVG-ray.py\
        --dataset femnist --model mlp --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 3 --test_freq 10 --frac 1 --repeat_id $i --description s1-flanp --hyper_setting noniid-hyper\
        --FEDAVG_local_ep 5 --FT_epoch 10 --flanp --leaf_path /home/lab304/zebang/dataset&
done
for i in {1..4}
do
    python LG-ray.py\
        --dataset femnist --model mlp --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 3 --test_freq 10 --frac 1 --repeat_id $i --description s1 --hyper_setting noniid-hyper\
        --LG_local_ep 5 --FT_epoch 10 --leaf_path /home/lab304/zebang/dataset&
done
for i in {1..4}
do
    python LG-ray.py\
        --dataset femnist --model mlp --num_classes 10 --epochs 150 --num_users 100 --gpu $(($i-1)) --seed $(($i * 10))\
        --shard_per_user 3 --test_freq 10 --frac 1 --repeat_id $i --description s1-flanp --hyper_setting noniid-hyper\
        --LG_local_ep 5 --FT_epoch 10 --flanp --leaf_path /home/lab304/zebang/dataset&
done
