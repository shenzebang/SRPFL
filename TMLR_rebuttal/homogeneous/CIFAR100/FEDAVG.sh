for i in {1..4}
do
    python FEDAVG-ray.py\
        --dataset cifar100 --model cnn --num_classes 100 --epochs 150 --num_users 100 --gpu 2 --seed $(($i * 10))\
        --shard_per_user 100 --test_freq 10 --frac 1 --repeat_id 1 --description s1 --hyper_setting iid-hyper\
        --FEDAVG_local_ep 5 --FT_epoch 10
done