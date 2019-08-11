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
            for CNeg in 1
            do
                for sigma in '[19.0,20.0,21.0,22.0,23.0]' #'[3.0,4.0,5.0,6.0,7.0]' '[1.0,5.0,10.0,20.0,30.0]'
                do
                    for alpha in 2.5
                    do
                        for trial_id in 0
                        do
                            for beta in 0.75
                            do
                                python2 gan_main.py --process_name RNS-p-p-R${Neg},${CNeg}-${dataset}-${alpha}-${beta}@dingjingtao --model RNS --gpu $gpu --dataset ${dataset}_click_data --regs $regs --num_neg 1 --embed_size 32 --batch_size 1024 --lr $lr --epochs 400 --optimizer Adam --trial_id ${trial_id} --verbose 1 --dis_model $dis_model --gen_model $gen_model --use_pretrain_dis --use_pretrain_gen --early_stop $early_stop --gen_file ${gen_file} --dis_file ${dis_file} --eval_pretrain --reduced --candidates $Neg --candidates_neg $CNeg --eval_mode $mode --LRecList $LRecList --alpha $alpha --sigma_range ${sigma} --beta ${beta} --save_model --early_stop_by_pretrain
                            done
                        done
                    done
                done
            done
        done
    done
done