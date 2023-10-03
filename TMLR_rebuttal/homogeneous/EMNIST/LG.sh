for i in {1..4}
do
    python LG-ray.py\
        --dataset emnist --model mlp --num_classes 10 --epochs 150 --num_users 100 --gpu 1 --seed $(($i * 10))\
        --shard_per_user 10 --test_freq 10 --frac 1 --repeat_id $i --description s1 --hyper_setting iid-hyper\
        --LG_local_ep 5 --FT_epoch 10
done