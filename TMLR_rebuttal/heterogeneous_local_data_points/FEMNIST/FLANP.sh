for i in {1..4}
do
    python FEDAVG-ray.py\
        --dataset femnist --model mlp --num_classes 10 --epochs 150 --num_users 100 --gpu 3 --seed $(($i * 10))\
        --shard_per_user 3 --test_freq 10 --frac 1 --repeat_id $i --description s1-flanp --hyper_setting iid-hyper\
        --FEDAVG_local_ep 5 --FT_epoch 10 --flanp
done