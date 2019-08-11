#!/bin/bash
gpu=0
dataset=zhihu
dis_model=GMF
dis_file=pretrain_model_dis_zhihu.pkl
mode=list
LRecList=160
early_stop=10
for lr in 0.001
do
    for regs in '[1e-5,1e-5]'
    do
        for DNS in 50
        do
            for id in 0
            do
                python2 gan_main.py --process_name DNS-p-K${DNS}-${lr}-${regs}-${dataset}@dingjingtao --model DNS --gpu $gpu --dataset ${dataset}_click_data --regs $regs --num_neg 1 --embed_size 32 --batch_size 1024 --lr $lr --epochs 250 --optimizer Adam --trial_id $id --verbose 1 --dis_model $dis_model --gen_model DNS --use_pretrain_dis --save_model --early_stop $early_stop --K_DNS $DNS --dis_file $dis_file --eval_pretrain --eval_mode $mode --LRecList $LRecList
            done
        done
    done
done