#!/bin/bash
gpu=0
dataset=zhihu
mode=list
LRecList=160

python2 gan_main.py --process_name ItemPop-${dataset}@dingjingtao --model ItemPop --gpu $gpu --dataset ${dataset}_click_data --batch_size 1024 --trial_id 0 --save_model --LRecList $LRecList --eval_mode $mode
mode=topK
topK=100
python2 gan_main.py --process_name ItemPop-${dataset}@dingjingtao --model ItemPop --gpu $gpu --dataset ${dataset}_click_data --batch_size 1024 --trial_id 0 --save_model --LRecList $LRecList --eval_mode $mode --topK $topK