for i in {1..4}
do
    python HFMAML-ray.py\
        --dataset cifar100 --model cnn --num_classes 100 --epochs 150 --num_users 100 --gpu 2 --seed $(($i * 10))\
        --shard_per_user 100 --test_freq 10 --frac 1 --repeat_id $i --description s1 --hyper_setting iid-hyper\
        --HFMAML_alpha 1e-2 --HFMAML_beta 1e-1 --HFMAML_delta 1e-3 --HFMAML_local_bs 10 --HFMAML_local_ep 5 --FT_epoch 10
done
