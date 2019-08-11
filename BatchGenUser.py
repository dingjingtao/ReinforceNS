import numpy as np

from multiprocessing import Pool
from multiprocessing import cpu_count
from time import time
import random

# np.random.seed(3)

_user_input = None
_item_input = None
_item_input_neg = None
_batch_size = None
_index = None



# input: dataset(Mat, List, Rating, Negatives), batch_choice, num_negatives
# output: [_user_input_list, _item_input_list, _labels_list]
def sampling(dataset, num_negatives, with_neg = [False], pretrain_g = False, pretrain_d = False):
    num_users, num_items = dataset.num_users, dataset.num_items
    # train
    if not pretrain_g and not pretrain_d:
        user_input, item_input, item_input_neg = [], [], []
        for u in dataset.trainDict:
            items = dataset.trainDict[u]
            for i in items:
                for _ in range(num_negatives):
                    user_input.append(u)
                    item_input.append(i)
                    item_input_neg.append(-1)
        return user_input, item_input, item_input_neg
    # pretrain-dis
    elif pretrain_d:
        user_input, item_input, item_input_neg = [], [], []
        for u in dataset.trainDict:
            items = dataset.trainDict[u]
            # uniform sampling non-click as negative
            if not with_neg[0]:
                for i in dataset.trainDictSet[u]:
                    for _ in range(num_negatives):
                        j = random.randint(0, num_items - 1)
                        while j in dataset.trainDictSet[u]:
                            j = random.randint(0, num_items - 1)
                        user_input.append(u)
                        item_input.append(i)
                        item_input_neg.append(j)
            # oversampling displayed as negative
            else:
                for i in items:
                    if u in dataset.trainNegDict:
                        for _ in range(num_negatives):
                            prob = random.random()
                            if with_neg[1] > prob:
                                j = np.random.choice(dataset.trainNegDictArray[u], 1)
                            else:
                                j = random.randint(0, num_items - 1)
                                while j in dataset.trainDictSet[u] or (u in dataset.trainNegDict and j in dataset.trainNegDict[u]):
                                    j = random.randint(0, num_items - 1)
                            user_input.append(u)
                            item_input.append(i)
                            item_input_neg.append(j)
                    else:
                        for _ in range(num_negatives):
                            j = random.randint(0, num_items - 1)
                            while j in dataset.trainDictSet[u]:
                                j = random.randint(0, num_items - 1)
                            user_input.append(u)
                            item_input.append(i)
                            item_input_neg.append(j)
        return user_input, item_input, item_input_neg



def shuffle(samples, batch_size, args):
    global _user_input
    global _item_input
    global _item_input_neg
    global _batch_size
    global _index


    _user_input, _item_input, _item_input_neg = samples
    _batch_size = batch_size
    _index = range(len(_user_input))
    np.random.shuffle(_index)
    num_batch = len(_user_input) // _batch_size
    if args.pretrain_gen:
        pool = Pool(10)
    else:
        pool = Pool(4)
    res = pool.map(_get_train_batch_BPR, range(num_batch))
    pool.close()
    pool.join()

    user_list = [r[0] for r in res]
    item_list = [r[1] for r in res]
    item_neg_list = [r[2] for r in res]
    return user_list, item_list, item_neg_list


def _get_train_batch_BPR(i):
    user_batch, item_batch, item_neg_batch = [], [], []
    begin = i * _batch_size
    for idx in range(begin, begin + _batch_size):
        user_batch.append(_user_input[_index[idx]])
        item_batch.append(_item_input[_index[idx]])
        item_neg_batch.append(_item_input_neg[_index[idx]])
    return np.array(user_batch), np.array(item_batch), np.array(item_neg_batch)


