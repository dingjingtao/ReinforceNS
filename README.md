# ReinforceNS
The implementation of RNS

RNS is a better negative sampler for recommendation with exposure data. This is our official implementation for the paper: 

Jingtao Ding, Yuhan Quan, Xiangnan He, Yong Li, Depeng Jin, **Reinforced Negative Sampling for Recommendation with Exposure Data**, In Proceedings of IJCAI'19.

Data is the Zhihu dataset in the paper.

Please run code with shell in 'sh/'

RNS:   bash rns.sh

KBGAN: bash kbgan.sh

DNS:   bash dns.sh

We have provided a pretrained model file for above methods.

BPR:   bash bpr.sh

ItemPop: bash itempop.sh

Two evaluation mode: topK or List (as the paper)
