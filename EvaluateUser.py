import math
import tensorflow as tf

from multiprocessing import Pool
import numpy as np
from time import time
import gc
import pdb
import scipy.io as sio

# Global variables that are shared across processes
_model = None
_sess = None
_dataset = None
_K = None
_DictList = None
_gtItem = None
_user_rating = None
_user_order = None
_model_name = None
_userIdList = None
_size_part = None

def init_evaluate_model(dataset):
    #DictList = []
    userIdList = []
    for idx in xrange(len(dataset.testRatings)):
        user, gtItem = dataset.testRatings[idx] # gtItem: [list of items]
        userIdList.append(user)
    return userIdList

def init_report_model(dataset):
    #DictList = []
    userIdList1 = []
    userIdList2 = []
    for idx in xrange(len(dataset.testRatings)):
        user, gtItem = dataset.testRatings[idx] # gtItem: [list of items]
        userIdList1.append(user)
    for idx in xrange(len(dataset.reportRatings)):
        user, gtItem = dataset.reportRatings[idx] # gtItem: [list of items]
        userIdList2.append(user)
    return userIdList1, userIdList2

def evalRecList(model, sess, dataset, userIdList, args, mode=True):
    global _model
    global _DictList
    global _sess
    global _dataset
    global _gtItem
    global _user_rating
    global _user_order
    global _model_name
    global _userIdList
    global eval_batch
    global _size_part

    cpu_num = 1
    eval_batch = len(dataset.testRatings) #2000
    if args.tensor_batch == 0:
        tensor_batch = eval_batch
    else:
        tensor_batch = args.tensor_batch
    print "start evaluate RecList"
    _dataset = dataset
    _model = model
    _sess = sess
    _model_name = args.model
    _userIdList = userIdList
    _gpus = len(args.gpu.split(','))
    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    aucs, ndcgs, _gtItem = [], [], []
    test_num = len(_dataset.testRatings)
    userids, itemids = [], []
    index = 0

    _user_rating = []
    _user_order = []
    eval_batch_begin = time()
    while True:

        UserList = []  # eval_batch*1

        if index > len(dataset.testRatings):
            break
        for idx in range(index, min(index + eval_batch, len(dataset.testRatings))):
            user, gtItem = dataset.testRatings[idx]  # gtItem: [list of items]
            #
            if idx == min(index + eval_batch, len(dataset.testRatings)) - 1:
                UserList.append([user])
                if _user_rating == []:
                    list_rating, list_order = _sess.run([_model.list_rating,_model.list_order], feed_dict={model.user_input: UserList, _model.candidates_reclist: dataset.testRecList[idx + 1 - len(UserList):idx + 1, :]})
                    _user_rating.extend(zip(UserList, list_rating))
                    _user_order.extend(list_order)
                else:
                    user_prediction_temp, user_order_temp = _sess.run([_model.list_rating,_model.list_order], feed_dict={model.user_input: UserList, _model.candidates_reclist: dataset.testRecList[idx+1-len(UserList):idx+1,:]})
                    _user_rating.extend(zip(UserList, user_prediction_temp))
                    _user_order.extend(user_order_temp)

            elif (idx - index) > 0 and (idx - index) % tensor_batch == 0:
                if _user_rating == []:
                    user_prediction_temp, user_order_temp = _sess.run([_model.list_rating,_model.list_order], feed_dict={model.user_input: UserList, _model.candidates_reclist: dataset.testRecList[idx-len(UserList):idx,:]})
                    _user_rating.extend(zip(UserList, user_prediction_temp))
                    _user_order.extend(user_order_temp)

                else:
                    user_prediction_temp, user_order_temp = _sess.run([_model.list_rating,_model.list_order], feed_dict={model.user_input: UserList, _model.candidates_reclist: dataset.testRecList[idx-len(UserList):idx,:]})
                    _user_rating.extend(zip(UserList, user_prediction_temp))
                    _user_order.extend(user_order_temp)
                UserList = [[user]]
            else:
                UserList.append([user])

            _gtItem.append(gtItem)  # append a [list of items]

        index += eval_batch
    eval_batch_time_0 = time() - eval_batch_begin
    print eval_batch_time_0
    _size_part = int(math.ceil(len(_user_rating) / (0. + cpu_num)))
    _user_order = np.array(_user_order)

    res = _eval_users_list(0)
    userids.extend(res[0])
    itemids.extend(res[1])
    aucs.extend(res[2])
    ndcgs.extend(res[3])
    eval_batch_time = time() - eval_batch_begin

    return (userids, itemids, aucs, ndcgs)

def evalReportRecList(model, sess, dataset, userIdList, args, mode=True):
    global _model
    global _DictList
    global _sess
    global _dataset
    global _gtItem
    global _user_rating
    global _user_order
    global _model_name
    global _userIdList
    global eval_batch
    global _size_part

    cpu_num = 1
    eval_batch = len(dataset.reportRatings) #2000
    if args.tensor_batch == 0:
        tensor_batch = eval_batch
    else:
        tensor_batch = args.tensor_batch
    print "start evaluate RecList"
    _dataset = dataset
    _model = model
    # _DictList = DictList
    _sess = sess
    _model_name = args.model
    _userIdList = userIdList
    _gpus = len(args.gpu.split(','))
    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    aucs, ndcgs, _gtItem = [], [], []

    test_num = len(_dataset.reportRatings)

    # give predictions on users
    # for idx in xrange(len(_DictList)):

    userids, itemids = [], []
    index = 0

    _user_rating = []
    _user_order = []
    eval_batch_begin = time()
    while True:

        UserList = []  # eval_batch*1

        if index > len(dataset.reportRatings):
            break
        for idx in range(index, min(index + eval_batch, len(dataset.reportRatings))):
            user, gtItem = dataset.reportRatings[idx]  # gtItem: [list of items]
            #
            if idx == min(index + eval_batch, len(dataset.reportRatings)) - 1:
                UserList.append([user])
                if _user_rating == []:
                    list_rating, list_order = _sess.run([_model.list_rating,_model.list_order], feed_dict={model.user_input: UserList, _model.candidates_reclist: dataset.reportRecList[idx + 1 - len(UserList):idx + 1, :]})
                    _user_rating.extend(zip(UserList, list_rating))
                    _user_order.extend(list_order)
                else:
                    user_prediction_temp, user_order_temp = _sess.run([_model.list_rating,_model.list_order], feed_dict={model.user_input: UserList, _model.candidates_reclist: dataset.reportRecList[idx+1-len(UserList):idx+1,:]})
                    _user_rating.extend(zip(UserList, user_prediction_temp))
                    _user_order.extend(user_order_temp)

            elif (idx - index) > 0 and (idx - index) % tensor_batch == 0:
                if _user_rating == []:
                    user_prediction_temp, user_order_temp = _sess.run([_model.list_rating,_model.list_order], feed_dict={model.user_input: UserList, _model.candidates_reclist: dataset.reportRecList[idx-len(UserList):idx,:]})
                    _user_rating.extend(zip(UserList, user_prediction_temp))
                    _user_order.extend(user_order_temp)

                else:
                    user_prediction_temp, user_order_temp = _sess.run([_model.list_rating,_model.list_order], feed_dict={model.user_input: UserList, _model.candidates_reclist: dataset.reportRecList[idx-len(UserList):idx,:]})
                    _user_rating.extend(zip(UserList, user_prediction_temp))
                    _user_order.extend(user_order_temp)
                UserList = [[user]]
            else:
                UserList.append([user])

            _gtItem.append(gtItem)  # append a [list of items]

        index += eval_batch
    eval_batch_time_0 = time() - eval_batch_begin
    print eval_batch_time_0
    _size_part = int(math.ceil(len(_user_rating) / (0. + cpu_num)))
    _user_order = np.array(_user_order)

    # pool = Pool(cpu_num)
    # res = pool.map(_eval_users_list, range(cpu_num))
    #
    # pool.close()
    # pool.join()
    # for r in res:
    #     userids.extend(r[0])
    #     itemids.extend(r[1])
    #     aucs.extend(r[2])
    #     ndcgs.extend(r[3])
    res = _eval_report_users_list(0)
    userids.extend(res[0])
    itemids.extend(res[1])
    aucs.extend(res[2])
    ndcgs.extend(res[3])


    # index += eval_batch
    eval_batch_time = time() - eval_batch_begin



    return (userids, itemids, aucs, ndcgs)

def eval(model, sess, dataset, userIdList, args, mode = True):
    global _model
    global _K
    global _DictList
    global _sess
    global _dataset
    global _gtItem
    # global _user_prediction # for memory
    global _model_name
    global _userIdList
    global eval_batch

    cpu_num = 10
    eval_batch=2000
    if args.tensor_batch == 0:
        tensor_batch = eval_batch
    else:
        tensor_batch = args.tensor_batch
    print "start evaluate"
    _dataset = dataset
    _model = model

    _sess = sess
    _K = args.topK
    _model_name = args.model
    _userIdList = userIdList
    _gpus = len(args.gpu.split(','))
    # run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)

    hits, ndcgs, _gtItem= [], [], []
    items = range(_dataset.num_items) # rank on all items
    item_input = np.array(items)[:, None]
    test_num = len(_dataset.testRatings)

    userids,itemids=[],[]
    index=0

    _user_prediction = []
    eval_batch_begin = time()
    while True:
        UserList = []  # eval_batch*1
        if index>len(dataset.testRatings):
            break
        for idx in range(index,min(index+eval_batch,len(dataset.testRatings))):
            user, gtItem = dataset.testRatings[idx] # gtItem: [list of items]
            #
            if idx == min(index + eval_batch, len(dataset.testRatings)) - 1:
                UserList.append([user])
                if _user_prediction == []:
                    _user_prediction.extend(zip( UserList, _sess.run(_model.all_rating, feed_dict={model.user_input: UserList})))
                else:
                    user_prediction_temp = _sess.run(_model.all_rating,
                                                                    feed_dict={model.user_input: UserList})
                    _user_prediction.extend(zip( UserList, user_prediction_temp))
                    # _user_prediction = np.concatenate([_user_prediction, user_prediction_temp], 0)
            elif (idx - index) > 0 and (idx - index) % tensor_batch == 0:
                if _user_prediction == []:
                    user_prediction_temp = _sess.run(_model.all_rating, feed_dict={model.user_input: UserList})
                    _user_prediction.extend(zip(UserList, user_prediction_temp))
                    # _user_prediction = np.concatenate(_user_prediction_tmp, 0)
                else:
                    user_prediction_temp = _sess.run(_model.all_rating,
                                  feed_dict={model.user_input: UserList})
                    # _user_prediction = np.concatenate([_user_prediction, user_prediction_temp], 0)
                    _user_prediction.extend(zip(UserList, user_prediction_temp))
                UserList = [[user]]
            else:
                UserList.append([user])

            _gtItem.append(gtItem) # append a [list of items]

        index += eval_batch
    eval_batch_time_0 = time() - eval_batch_begin
    print eval_batch_time_0
    pool = Pool(cpu_num)
    res = pool.map(_eval_one_user_true, _user_prediction)
    pool.close()
    pool.join()

    userids = userids + [r[0] for r in res]
    itemids = itemids + [r[1] for r in res]
    hits = hits + [r[2] for r in res]
    ndcgs = ndcgs + [r[3] for r in res]
    eval_batch_time = time() - eval_batch_begin

    del _user_prediction
    gc.collect()

    return (userids, itemids, hits, ndcgs)

def _eval_report_users_list(i):
    L = len(_user_rating)
    user_rating = _user_rating[i*_size_part: min((i+1)*_size_part, L)]
    user_order = _user_order[i*_size_part: min((i+1)*_size_part, L),:] # np.narray
    user_flag = _dataset.reportFlagList[i*_size_part: min((i+1)*_size_part, L),:] # np.narray

    dcg = np.sum(np.log(2) * ((np.log(user_order+2))**(-1)) * user_flag,axis=1) # (n,)
    M, N = np.shape(user_order)
    dcg_max = np.sum(np.log(2) * ((np.log(np.tile(range(N-1+2,-1+2,-1),[M,1])))**(-1)) * user_flag,axis=1) # (n,)
    ndcg = dcg / dcg_max

    user_order_masked = (user_order+1) * user_flag
    user_order_insideClicked = np.argsort(np.argsort(-user_order_masked, axis=1), axis=1)
    anti_auc = np.sum((user_order - user_order_insideClicked) * user_flag, axis=1)
    auc_max = (N-np.sum(user_flag, axis=1)) * np.sum(user_flag, axis=1)
    auc = (auc_max-anti_auc) / (0.+auc_max)

    gtItem = _gtItem[i*_size_part: min((i+1)*_size_part, L)] # [list of pos items]
    userid = _userIdList[i*_size_part: min((i+1)*_size_part, L)]

    return (userid, gtItem, auc, ndcg)

def _eval_users_list(i):
    L = len(_user_rating)
    user_rating = _user_rating[i*_size_part: min((i+1)*_size_part, L)]
    user_order = _user_order[i*_size_part: min((i+1)*_size_part, L),:] # np.narray
    user_flag = _dataset.testFlagList[i*_size_part: min((i+1)*_size_part, L),:] # np.narray

    dcg = np.sum(np.log(2) * ((np.log(user_order+2))**(-1)) * user_flag,axis=1) # (n,)
    M, N = np.shape(user_order)
    dcg_max = np.sum(np.log(2) * ((np.log(np.tile(range(N-1+2,-1+2,-1),[M,1])))**(-1)) * user_flag,axis=1) # (n,)
    ndcg = dcg / dcg_max

    user_order_masked = (user_order+1) * user_flag
    user_order_insideClicked = np.argsort(np.argsort(-user_order_masked, axis=1), axis=1)
    anti_auc = np.sum((user_order - user_order_insideClicked) * user_flag, axis=1)
    auc_max = (N-np.sum(user_flag, axis=1)) * np.sum(user_flag, axis=1)
    auc = (auc_max-anti_auc) / (0.+auc_max)

    gtItem = _gtItem[i*_size_part: min((i+1)*_size_part, L)] # [list of pos items]
    userid = _userIdList[i*_size_part: min((i+1)*_size_part, L)]

    return (userid, gtItem, auc, ndcg)

def _eval_one_user_true(_user_prediction):
    # predictions = _user_prediction[idx%eval_batch]
    idx = _user_prediction[0][0]
    predictions = _user_prediction[1]
    gtItem = _gtItem[idx] # [list of pos items]
    userid = _userIdList[idx]


    rank_score = predictions[gtItem]
    rank = np.zeros((len(gtItem),1),dtype=np.int32)

    cur = 0
    for i in predictions:
        early_stop = 0
        for s in xrange(len(gtItem)):
            if i >= rank_score[s] and gtItem[s] != cur:
                rank[s] += 1
            if rank[s] >= _K:
                early_stop += 1
        if early_stop == len(gtItem):
            hr = 0
            ndcg = 0
            return (userid, gtItem, hr, ndcg)
        cur += 1
    hr = 0
    dcg_max = 0
    dcg = 0
    for s in xrange(len(gtItem)):
        if rank[s] < _K:
            hr += 1
            dcg += math.log(2) / math.log(rank[s] + 2)
        dcg_max += math.log(2) / math.log(s + 2)
    hr /= (0.+len(gtItem))
    ndcg = dcg/dcg_max

    return (userid, gtItem, hr, ndcg)