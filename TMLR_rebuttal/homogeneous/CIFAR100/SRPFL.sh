for i in {1..4}
do
    python FEDREP-ray.py\
        --dataset cifar100 --model cnn --num_classes 100 --epochs 150 --num_users 100 --gpu 2 --seed $(($i * 10))\
        --shard_per_user 100 --test_freq 10 --frac 1 --repeat_id $i --description s1-flanp --hyper_setting iid-hyper\
        --FEDREP_local_rep_ep 5 --FEDREP_head_ep_per_rep_update 10 --FT_epoch 10 --flanp
done