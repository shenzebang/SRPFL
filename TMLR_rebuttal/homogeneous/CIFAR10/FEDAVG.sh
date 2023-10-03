for i in {1..4}
do
    python FEDAVG-ray.py\
        --dataset cifar10 --model cnn --num_classes 10 --epochs 150 --num_users 100 --gpu 0 --seed $(($i * 10))\
        --shard_per_user 10 --test_freq 10 --frac 1 --repeat_id $i --description s1 --hyper_setting iid-hyper\
        --FEDAVG_local_ep 1 --FT_epoch 10
done