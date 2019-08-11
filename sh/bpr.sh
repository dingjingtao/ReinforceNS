#!/bin/bash
gpu=0
dataset=zhihu
dis_model=GMF
mode=list
LRecList=160
early_stop=10
for embed in 32
do
    for lr in 1e-3
    do
        for regs in '[1e-5,1e-5]'
        do
            for id in 0
            do
                python2 gan_main.py --process_name ${dis_model}-${dataset}-${lr}-${regs}-F${embed}@dingjingtao --model RNS --gpu $gpu --dataset ${dataset}_click_data --regs $regs --num_neg 4 --embed_size $embed --batch_size 1024 --lr $lr --epochs 250 --optimizer Adam --trial_id $id --verbose 1 --dis_model $dis_model --pretrain_dis --save_model --early_stop $early_stop --LRecList $LRecList --eval_mode $mode
            done
        done
    done
done