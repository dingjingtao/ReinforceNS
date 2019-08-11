#!/bin/bash
gpu=0
dataset=zhihu
dis_model=GMF
gen_model=GMF
gen_file=pretrain_model_dis_zhihu.pkl
dis_file=pretrain_model_dis_zhihu.pkl
mode=list
LRecList=160
early_stop=10
for lr in 0.001
do
    for regs in '[1e-5,1e-5]'
    do
        for Neg in 30
        do
            for id in 0
            do
                python2 gan_main.py --process_name KBGAN-p-p-R${Neg}-${lr}-${regs}@dingjingtao --model KBGAN --gpu $gpu --dataset ${dataset}_click_data --regs $regs --num_neg 1 --embed_size 32 --batch_size 1024 --lr $lr --epochs 400 --optimizer Adam --trial_id $id --verbose 1 --use_pretrain_dis --use_pretrain_gen --eval_pretrain --dis_file $dis_file --gen_file $gen_file --dis_model $dis_model --gen_model $gen_model --early_stop $early_stop --reduced --candidates $Neg --eval_mode $mode --LRecList $LRecList --save_model
            done
        done
    done
done