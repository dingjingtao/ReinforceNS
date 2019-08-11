from __future__ import division

import os
import tensorflow as tf
import numpy as np
import math
import random
import logging

from time import time, sleep

from baseline_model import ItemPop
from Generator import GEN, DNS
from Discriminator import DIS

import BatchGenUser as BatchUser
import EvaluateUser as EvalUser
from dataset import Dataset

import argparse
from scipy import sparse
from scipy import io as sio

import setproctitle

from multiprocessing import Process, cpu_count, Semaphore, Lock, Pool, Queue, JoinableQueue

import cPickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


Log_dir_name = None
Result_dir_name = None
Param_dir_name = None
Model_dir_name = None

def parse_args():
    parser = argparse.ArgumentParser(description="Run Sampler-GAN.")
    parser.add_argument('--dataset', nargs='?', default='zhihu',
                        help='Choose a dataset: zhihu')
    parser.add_argument('--model', nargs='?', default='BPR',
                        help='Choose model: BPR, ItemPop')
    parser.add_argument('--eval_mode', nargs='?', default='topK',
                        help='Choose mode: topK, list, report')
    parser.add_argument('--loss_func', nargs='?', default='BPR',
                        help='Choose loss: logloss, BPR')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2048,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs.')
    parser.add_argument('--embed_size', type=int, default=32,
                        help='Embedding size.')
    parser.add_argument('--layer_num', type=int, default=0,
                        help='MLP layer num.')
    parser.add_argument('--regs', nargs='?', default='[0.0,0.0]',
                        help='Regularization for user and item embeddings. DIS-GEN')
    parser.add_argument('--reg_mode', type=bool, default=True,
                        help='Reg Mode')
    parser.add_argument('--train_loss', type=bool, default=True,
                        help='Calculate training loss or not')
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')
    parser.add_argument('--process_name', nargs='?', default='SamplerGAN-tf@dingjingtao',
                        help='Input process name.')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU.')
    parser.add_argument('--evaluate', nargs='?', default='yes',
                        help='Evaluate or not.')
    parser.add_argument('--tensor_batch', type=int, default=0,
                        help='batch size for all-rating calculation when evaluating.')
    parser.add_argument('--optimizer', nargs='?', default='GradientDescent',
                        help='Choose an optimizer: GradientDescent, Adagrad, Adam')
    parser.add_argument('--plot_network', nargs='?', type=bool, default=False,
                        help='Dir to store tensorboard log')
    parser.add_argument('--topK', nargs='?', type=int, default=100,
                        help='topK for hr/ndcg')
    parser.add_argument('--multiprocess', nargs='?', default='yes',
                        help='Evaluate multiprocessingly or not')
    parser.add_argument('--trial_id', nargs='?', default='',
                        help='Indicate trail id with same condition')
    parser.add_argument('--dropout', type=float, default=1.0,
                        help='dropout keep_prob')
    parser.add_argument('--save_param', action='store_true', default=False,
                        help='Save param or not, related to args.verbose')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help='Save final model or not')
    parser.add_argument('--load_param', type=str, default=None,
                        help='Path where para is saved')
    parser.add_argument('--alpha', type=float, default=0.00,
                        help='alpha')
    parser.add_argument('--temperature', type=float, default=1.00,
                        help='temperature')
    parser.add_argument('--use_pretrain_dis', action='store_true', default=False,
                        help='use pre-train model or not')
    parser.add_argument('--use_pretrain_gen', action='store_true', default=False,
                        help='use pre-train model or not')
    parser.add_argument('--dis_model', type=str, default="GMF",
                        help='DIS model')
    parser.add_argument('--gen_model', type=str, default="GMF",
                        help='GEN model')
    parser.add_argument('--pretrain_dis', action='store_true', default=False,
                        help='pre-train model or not')
    parser.add_argument('--pretrain_gen', action='store_true', default=False,
                        help='pre-train model or not')
    parser.add_argument('--eval_pretrain', action='store_true', default=False,
                        help='eval pre-train-model or not')
    parser.add_argument('--debug', action='store_true', default=False,
                        help='debug or not')
    parser.add_argument('--early_stop', nargs='?', type=int, default=5,
                        help='early_stop evals')
    parser.add_argument('--dns', action='store_true', default=False,
                        help='pretrain generator for dns or not')
    parser.add_argument('--eval2', type=int, default=0,
                        help='epoch after which use the 2nd eval setting.')
    parser.add_argument('--verbose2', type=int, default=1,
                        help='Interval of evaluation2.')
    parser.add_argument('--gen_file', type=str, default="pretrain_model_gen.pkl",
                        help='Pretrained GEN model file')
    parser.add_argument('--dis_file', type=str, default="pretrain_model_dis.pkl",
                        help='Pretrained DIS model file')
    parser.add_argument('--K_DNS', type=int, default=1,
                        help='candidate number K for DNS.')
    parser.add_argument('--candidates', type=int, default=0,
                        help='number for negative candidate set.')
    parser.add_argument('--reduced', action='store_true', default=False,
                        help='reduced sampling for negative samples')
    parser.add_argument('--lr_g', type=float, default=-1,
                        help='Learning rate for gen.')
    parser.add_argument('--candidates_neg', type=int, default=0,
                        help='number for negative candidate set (Neg).')
    parser.add_argument('--LRecList', type=int, default=1,
                        help='RecList Length.')
    parser.add_argument('--write_list', action='store_true', default=False,
                        help='wirite RecList file')
    parser.add_argument('--neg_samples_only',
                        action='store_true', default=False, help='whether to oversample displayed instances as negative')
    parser.add_argument('--neg_samples_ratio', type=float, default=1.0, help='neg_samples_ratio')
    parser.add_argument('--c_entropy', type=float, default=1.0,
                        help='c for entropy log(c).')
    parser.add_argument('--sigma_range', nargs='?', default='[1.0,5.0,10.0,20.0,30.0]',
                        help='sigma range for gaussian kernals.')
    parser.add_argument('--beta', type=float, default=0.0,
                        help='beta')
    parser.add_argument('--NoDNSLoss',
                        action='store_true', default=False, help='Not include the DNS Loss')
    parser.add_argument('--eval_by_ndcg',
                        type=bool, default=False, help='eval by ndcg')
    parser.add_argument('--early_stop_by_pretrain',
                        action='store_true', default=False, help='early stop by pretrain')

    return parser.parse_args()

def pretraining(model, train, args, base_epoch=0):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    global pool
    global eval_queue, job_num, loss_list
    global dataset_cur, eval_queue, job_num
    global hr_list, ndcg_list

    # initialize for Evaluate
    if train == 'dis':
        userIdList = EvalUser.init_evaluate_model(dataset_cur)
    else:
        print ("No pretrain-gen mode!!! -> exit")
        exit(0)


    with tf.Session(config=config) as sess:
        # plot network
        # tensorboard
        print('plot_network:', args.plot_network)
        if args.plot_network:
            print "writing network to", args.plot_network
            writer = tf.summary.FileWriter(Log_dir_name + '/' + filename + '-tensorboard', graph=sess.graph)

            with tf.name_scope("eval"):
                if args.eval_mode == 'topK':
                    hr_tf = tf.placeholder(tf.float32)
                    ndcg_tf = tf.placeholder(tf.float32)

                    tf.summary.scalar('HR', hr_tf)
                    tf.summary.scalar('NDCG', ndcg_tf)
                elif args.eval_mode == 'list':
                    hr_tf = tf.placeholder(tf.float32)
                    ndcg_tf = tf.placeholder(tf.float32)

                    tf.summary.scalar('AUC', hr_tf)
                    tf.summary.scalar('NDCG', ndcg_tf)

            with tf.name_scope("training"):
                train_loss_tf = tf.placeholder(tf.float32)
                tf.summary.scalar('train_loss', train_loss_tf)

            merged = tf.summary.merge_all()

        # initial training
        sess.run(tf.global_variables_initializer())
        logging.info("--- Start Pre-training %s ---" %train)
        print("--- Start Pre-training %s ---" %train)

        train_loss = 0
        train_time = 0
        batch_time = 0

        stop_counter = 0

        eval_begin = time()
        if train == 'dis':
            hr_best = 0
            if args.eval_pretrain:
                if args.eval_mode=="topK":
                    userids, itemids, hits, ndcgs = EvalUser.eval(model, sess, dataset_cur, userIdList, args)
                elif args.eval_mode=="list":
                    userids, itemids, hits, ndcgs = EvalUser.evalRecList(model, sess, dataset_cur, userIdList, args)
                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                eval_time = time() - eval_begin
                print ("DIS: Pre-Train[ %.1f ] %.4f %.4f" % (eval_time, hr, ndcg))

        # finalize the graph
        tf.get_default_graph().finalize()
        # train by epoch
        for epoch_count in range(args.epochs):
            # Batch
            batch_begin = time()
            if train == 'dis':
                samples = BatchUser.sampling(dataset_cur, args.num_neg, pretrain_d=True, with_neg=[args.neg_samples_only, args.neg_samples_ratio])

            batches = BatchUser.shuffle(samples, args.batch_size, args)
            batch_time = time() - batch_begin

            # Pre-Training
            train_begin = time()
            train_loss = pretraining_batch(model, sess, batches, args)
            train_time = time() - train_begin

            # Evaluate
            eval_flag = False
            if args.evaluate == 'yes':
                if args.eval2 == 0 or epoch_count <= args.eval2:
                    if epoch_count % args.verbose == 0:
                        eval_flag = True
                else:
                    if (epoch_count-args.eval2) % args.verbose2 == 0:
                        eval_flag = True

            # eval current epoch
            if eval_flag:
                eval_begin = time()
                if train == 'dis':
                    if args.eval_mode == "topK":
                        userids, itemids, hits, ndcgs = EvalUser.eval(model, sess, dataset_cur, userIdList, args)
                        metric1 = "HR"
                    elif args.eval_mode == "list":
                        userids, itemids, hits, ndcgs = EvalUser.evalRecList(model, sess, dataset_cur,
                                                                             userIdList, args)
                        metric1 = "AUC"
                    assert (len(userids) == len(hits) == len(ndcgs))
                    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                    eval_time = time() - eval_begin

                    logging.info(
                        "Epoch %d [%.1fs + %.1fs]: train_loss = %.4f  [%.1fs] %s = %.4f, NDCG = %.4f"
                        % (epoch_count + 1, batch_time, train_time, train_loss, eval_time, metric1, hr, ndcg))
                    print "Epoch %d [%.1fs + %.1fs]: train_loss = %.4f  [%.1fs] %s = %.4f, NDCG = %.4f" \
                          % (epoch_count + 1, batch_time, train_time, train_loss, eval_time, metric1, hr, ndcg)

                    # save results, save model
                    hr_list[epoch_count + base_epoch], ndcg_list[epoch_count + base_epoch], \
                    loss_list[epoch_count + base_epoch] = (hr, ndcg, train_loss)

                    if not args.eval_by_ndcg:
                        metric_cur = hr
                    else:
                        metric_cur = ndcg

                    if metric_cur > hr_best:
                        stop_counter = 0
                        hr_best = metric_cur
                        if args.save_model:
                            if train == 'dis':
                                save_path = Model_dir_name + '/' + args.model + '/Pretrain/DIS/' + model_filename + '/' + 'pretrain_model_dis.pkl'
                            else:
                                save_path = Model_dir_name + '/' + args.model + '/Pretrain/GEN/' + model_filename + '/' + 'pretrain_model_gen.pkl'
                            model.save_model(sess, save_path)
                            logging.info(
                                "Epoch %d [%.1fs + %.1fs]: train_loss = %.4f  [%.1fs] %s = %.4f, NDCG = %.4f, Model Saved"
                                % (epoch_count + 1, batch_time, train_time, train_loss, eval_time, metric1, hr, ndcg))
                            print('Model saved in ' + save_path)
                    else:
                        stop_counter += 1
                        if stop_counter > args.early_stop:
                            print ("early stopped")
                            logging.info("early stopped")
                            exit(0)
                    # tensorboard
                    if args.plot_network:
                        result = sess.run(merged, feed_dict={
                            train_loss_tf: train_loss,
                            hr_tf: hr,
                            ndcg_tf: ndcg
                        })
                        writer.add_summary(result, epoch_count + 1)
            # do not eval current epoch
            else:
                logging.info("Epoch %d [%.1fs + %.1fs]: train_loss = %.4f"
                             % (epoch_count + 1, batch_time, train_time, train_loss))
                print "Epoch %d [%.1fs + %.1fs]: train_loss = %.4f" \
                      % (epoch_count + 1, batch_time, train_time, train_loss)
                # tensorboard
                if args.plot_network:
                    result = sess.run(merged, feed_dict={
                        train_loss_tf: train_loss
                    })
                    writer.add_summary(result, epoch_count + 1)

def reportModelMetrics(dis, args):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1
    global pool
    global eval_queue, job_num
    global dataset_cur, eval_queue, job_num

    # initialize for Evaluate
    userIdList1, userIdList2 = EvalUser.init_report_model(dataset_cur)
    with tf.Session(config=config) as sess:

        # initial training
        sess.run(tf.global_variables_initializer())
        logging.info("--- Start Report ---")
        print("--- Start Report ---")

        _, _, hits1, ndcgs1 = EvalUser.evalRecList(dis, sess, dataset_cur, userIdList1, args)
        print("Validation: %.4f, %.4f" %( np.array(hits1).mean(), np.array(ndcgs1).mean()))
        _, _, hits2, ndcgs2 = EvalUser.evalReportRecList(dis, sess, dataset_cur, userIdList2, args)
        print("Test: %.4f, %.4f" % (np.array(hits2).mean(), np.array(ndcgs2).mean()))

def training(dis, gen, args, base_epoch=0):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.1

    global pool
    global eval_queue, job_num, loss_dis_list, loss_gen_list
    global dataset_cur, eval_queue, job_num
    global hr_list, ndcg_list

    # initialize for Evaluate
    userIdList = EvalUser.init_evaluate_model(dataset_cur)

    with tf.Session(config=config) as sess:
        # plot network
        # tensorboard
        print('plot_network:', args.plot_network)
        if args.plot_network:
            print "writing network to", args.plot_network
            writer = tf.summary.FileWriter(Log_dir_name + '/' + filename + '-tensorboard', graph=sess.graph)

            with tf.name_scope("eval"):
                if args.eval_mode == 'topK':
                    hr_tf = tf.placeholder(tf.float32)
                    ndcg_tf = tf.placeholder(tf.float32)

                    tf.summary.scalar('HR', hr_tf)
                    tf.summary.scalar('NDCG', ndcg_tf)
                elif args.eval_mode == 'list':
                    hr_tf = tf.placeholder(tf.float32)
                    ndcg_tf = tf.placeholder(tf.float32)

                    tf.summary.scalar('AUC', hr_tf)
                    tf.summary.scalar('NDCG', ndcg_tf)

            with tf.name_scope("training"):
                train_loss_tf = tf.placeholder(tf.float32)
                tf.summary.scalar('train_loss', train_loss_tf)

            merged = tf.summary.merge_all()

        # initial training
        sess.run(tf.global_variables_initializer())
        logging.info("--- Start training ---")
        print("--- Start training ---")

        if args.model == 'ItemPop' and args.evaluate == 'yes':
            eval_begin = time()
            if args.eval_mode == "topK":
                userids, itemids, hits, ndcgs = EvalUser.eval(dis, sess, dataset_cur, userIdList, args)
                metric1 = "HR"
            elif args.eval_mode == "list":
                userids, itemids, hits, ndcgs = EvalUser.evalRecList(dis, sess, dataset_cur, userIdList, args)
                metric1 = "AUC"
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            eval_time = time() - eval_begin
            logging.info("[%.1fs] %s = %.4f, NDCG = %.4f" % (eval_time, metric1, hr, ndcg))
            print "[%.1fs] %s = %.4f, NDCG = %.4f" % (eval_time, metric1, hr, ndcg)

            return

        reward_average = 0
        reward_realNeg_average = 0

        hr_best = 0
        stop_counter = 0

        if args.eval_pretrain:
            if args.eval_mode == "topK":
                userids, itemids, hits, ndcgs = EvalUser.eval(dis, sess, dataset_cur, userIdList, args)
            elif args.eval_mode == "list":
                userids, itemids, hits, ndcgs = EvalUser.evalRecList(dis, sess, dataset_cur, userIdList, args)
            hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
            print ("DIS: Pre-Train %.4f %.4f" % (hr, ndcg))
            hr_pretrain = hr
        else:
            hr_pretrain = 0
        # train by epoch
        for epoch_count in range(args.epochs):
            # Batch
            batch_begin = time()
            samples = BatchUser.sampling(dataset_cur, args.num_neg)

            batches = BatchUser.shuffle(samples, args.batch_size, args)
            batch_time = time() - batch_begin

            # Training
            train_begin = time()
            if args.model == 'DNS':
                loss_dis_average, loss_gen_average = training_dns_batch(dis, gen, sess, batches, args, epoch_count)
            else: # KBGAN/RNS
                loss_dis_average, loss_gen_average, reward_average, reward_realNeg_average = \
                    training_gan_batch(dis, gen, reward_average, reward_realNeg_average, sess, batches, args)
            train_time = time() - train_begin

            if np.isnan(loss_gen_average) or np.isnan(loss_dis_average):
                print "Loss NAN"
                logging.info("Loss NAN")
                exit(0)
            # Evaluate
            eval_flag = False
            if args.evaluate == 'yes':
                if args.eval2 == 0 or epoch_count <= args.eval2:
                    if epoch_count % args.verbose == 0:
                        eval_flag = True
                else:
                    if (epoch_count - args.eval2) % args.verbose2 == 0:
                        eval_flag = True

            # eval current epoch
            if eval_flag:
                eval_begin = time()
                if args.eval_mode == 'topK':
                    userids,itemids,hits,ndcgs=EvalUser.eval(dis, sess, dataset_cur, userIdList, args)
                    metric1 = "HR"
                elif args.eval_mode == 'list':
                    userids, itemids, hits, ndcgs = EvalUser.evalRecList(dis, sess, dataset_cur, userIdList, args)
                    metric1 = "AUC"
                assert (len(userids) == len(hits) == len(ndcgs))

                hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
                eval_time = time() - eval_begin

                logging.info("Epoch %d [%.1fs + %.1fs]: dis_loss = %.4f  gen_loss = %.4f  [%.1fs] %s = %.4f, NDCG = %.4f"
                    % (epoch_count + 1, batch_time, train_time, loss_dis_average, loss_gen_average, eval_time, metric1, hr, ndcg))
                print "Epoch %d [%.1fs + %.1fs]: dis_loss = %.4f  gen_loss = %.4f  [%.1fs] %s = %.4f, NDCG = %.4f" \
                      % (epoch_count + 1, batch_time, train_time, loss_dis_average, loss_gen_average, eval_time, metric1, hr, ndcg)

                # save results, save model
                hr_list[epoch_count + base_epoch], ndcg_list[epoch_count + base_epoch], \
                loss_dis_list[epoch_count + base_epoch], loss_gen_list[epoch_count + base_epoch] = (
                    hr, ndcg, loss_dis_average, loss_gen_average)

                if not args.eval_by_ndcg:
                    metric_cur = hr
                else:
                    metric_cur = ndcg

                if metric_cur > hr_best:
                    stop_counter = 0
                    hr_best = metric_cur
                    if args.save_model:
                        save_path_dis = Model_dir_name + '/' + args.model + '/' + model_filename + '/' + 'dis_models.pkl'
                        dis.save_model(sess, save_path_dis)
                        save_path_gen = Model_dir_name + '/' + args.model + '/' + model_filename + '/' + 'gen_models.pkl'
                        if args.model != 'DNS':
                            gen.save_model(sess, save_path_gen)
                        logging.info(
                            "Epoch %d [%.1fs + %.1fs]: dis_loss = %.4f  gen_loss = %.4f  [%.1fs] %s = %.4f, NDCG = %.4f, Model Saved"
                            % (
                            epoch_count + 1, batch_time, train_time, loss_dis_average, loss_gen_average, eval_time, metric1, hr, ndcg))
                        print('Model saved in ' + save_path_dis + ' and ' + save_path_gen)
                else:
                    stop_counter += 1
                    if stop_counter > args.early_stop:
                        print ("early stopped")
                        logging.info("early stopped")
                        exit(0)
                if args.early_stop_by_pretrain and (not args.pretrain_gen) and (not args.pretrain_dis) and hr < hr_pretrain and epoch_count >= 10:
                    print ("early stopped - worse than pretrain after 10 epochs")
                    logging.info("early stopped - worse than pretrain after 10 epochs")
                    exit(0)
                # tensorboard
                if args.plot_network:
                    result = sess.run(merged, feed_dict={
                                                         train_loss_tf: [loss_dis_average,loss_gen_average],
                                                         hr_tf: hr,
                                                         ndcg_tf: ndcg
                                                         })
                    writer.add_summary(result, epoch_count + 1)
            # do not eval current epoch
            else:
                logging.info("Epoch %d [%.1fs + %.1fs]: dis_loss = %.4f  gen_loss = %.4f"
                             % (epoch_count + 1, batch_time, train_time, loss_dis_average, loss_gen_average))
                print "Epoch %d [%.1fs + %.1fs]: dis_loss = %.4f  gen_loss = %.4f" \
                      % (epoch_count + 1, batch_time, train_time, loss_dis_average, loss_gen_average)
                # tensorboard
                if args.plot_network:
                    result = sess.run(merged, feed_dict={train_loss_tf: [loss_dis_average, loss_gen_average]})
                    writer.add_summary(result, epoch_count + 1)

            # save param
            if epoch_count % args.verbose == 0 and args.save_param:
                saver = tf.train.Saver()
                save_path = saver.save(sess, Param_dir_name + '/' + args.model +
                                       '/' + param_filename + '/' + 'params.ckpt',
                                       write_meta_graph=False, global_step=epoch_count)
                print('Param saved in ' + save_path)

def training_dns_batch(dis, gen, sess, batches, args, epoch_cur):

    num_batch = len(batches[1])
    loss_dis_average = 0.0

    epoch_num = 0

    user_input, item_input = batches[0],batches[1]
    begin0 = time()

    if args.K_DNS >0:
        for i in range(len(user_input)):
            # for each minibatch
            # time
            begin = time() * 1000
            train1 = time() * 1000
            # Step 1: GEN sample negative items
            negsamples_candidates_K = np.random.choice(num_items,[len(user_input[i]),args.K_DNS])
            u_id = 0
            for u in user_input[i]:
                epoch_num += 1
                l = 0
                for k in negsamples_candidates_K[u_id]:
                    if k in dataset_cur.trainDictSet[u]:
                        negsamples_candidates_K[u_id, l] = random.randint(0, num_items - 1)
                        while negsamples_candidates_K[u_id, l] in dataset_cur.trainDictSet[u]:
                            negsamples_candidates_K[u_id, l] = random.randint(0, num_items - 1)
                    l += 1
                u_id += 1
            train2 = time() * 1000

            # Step 2: GEN get ratings of those sampled negative items
            user_input_K = np.tile(np.reshape(user_input[i], [-1,1]), [1, args.K_DNS]) # user_input[i] is a 1*n array
            feed_dict = {dis.user_input: np.reshape(user_input_K, [-1, 1]),
                         dis.item_input: np.reshape(negsamples_candidates_K, [-1, 1])}
            ratings_candidates_K = np.reshape(sess.run(dis.output, feed_dict), [-1, args.K_DNS])

            if epoch_cur > 0 or args.K_DNS < 16:
                negsamples = negsamples_candidates_K[range(len(user_input[i])), np.argmax(ratings_candidates_K, axis=1)]
            else:
                negsamples = negsamples_candidates_K[:, 0]
            # time
            train3 = time() * 1000

            # Step 3: Get reward from DIS, Update DIS
            feed_dict = {dis.user_input: np.reshape(user_input[i], [-1, 1]),
                         dis.item_input: np.reshape(item_input[i], [-1, 1]),
                         dis.item_input_neg: np.reshape(negsamples, [-1, 1])}
            _, loss_dis = sess.run([dis.optimizer, dis.loss], feed_dict)

            loss_dis_average += loss_dis
            # time
            train4 = time() * 1000
    # use whole item space as the candidate for choosing hard negatives (Do not work)
    else:
        for i in range(len(user_input)):
            feed_dict = {dis.user_input: np.reshape(user_input[i], [-1, 1])}
            all_rating = sess.run(dis.all_rating, feed_dict)

            x = []
            y = []
            u_id = 0
            for u in user_input[i]:
                epoch_num += 1
                x.extend([u_id] * len(dataset_cur.trainDict[u]))
                y.extend(dataset_cur.trainDict[u])
                u_id += 1
            all_rating[x,y] = 0.0

            negsamples = np.argmax(all_rating, axis=1)

            feed_dict = {dis.user_input: np.reshape(user_input[i], [-1, 1]),
                         dis.item_input: np.reshape(item_input[i], [-1, 1]),
                         dis.item_input_neg: np.reshape(negsamples, [-1, 1])}
            _, loss_dis = sess.run([dis.optimizer, dis.loss], feed_dict)
            loss_dis_average += loss_dis

    end0 = time()

    loss_dis_average /= epoch_num

    return loss_dis_average, 0.0

def training_gan_batch(dis, gen, reward_average, reward_realNeg_average, sess, batches, args):

    num_batch = len(batches[1])
    loss_dis_average = 0.0
    loss_gen_average = 0.0

    reward_epoch = 0.0
    reward_realNeg_epoch=0.0
    epoch_num = 0
    cnt_realNeg = 0
    reward_realNegId_epoch = 0.0
    reward_realNegFeature_epoch = 0.0

    user_input, item_input = batches[0],batches[1]

    begin0 = time()

    for i in range(len(user_input)):
        # for each minibatch

        # time
        begin = time() * 1000

        # Step 1: GEN generate sampling space
        # KBGAN/RNS No Reduced Sampling space (Do not work)
        # use the whole item space as the candidate set for negative sampling in generator
        if not args.reduced:
            u_id = 0
            user_i_pos = []
            for u in user_input[i]:
                user_i_pos.extend(zip([u_id] * len(dataset_cur.trainDict[u]), dataset_cur.trainDict[u]))
                u_id += 1
                # training records num
                epoch_num += 1
        # KBGAN/RNS with Reduced Sampling space (Recommended)
        else:
            # KBGAN
            if args.model == 'KBGAN' or args.candidates_neg==-1: # KBGAN/CNeg==-1
                negsamples_candidates_K = np.random.choice(num_items, [len(user_input[i]), args.candidates])
                u_id = 0
                for u in user_input[i]:
                    epoch_num += 1
                    l = 0
                    for k in negsamples_candidates_K[u_id]:
                        if k in dataset_cur.trainDictSet[u]:
                            negsamples_candidates_K[u_id, l] = random.randint(0, num_items - 1)
                            while negsamples_candidates_K[u_id, l] in dataset_cur.trainDictSet[u]:
                                negsamples_candidates_K[u_id, l] = random.randint(0, num_items - 1)
                        l += 1
                    u_id += 1
            # RNS
            else:
                negsamples_candidates_K1 = np.random.choice(num_items, [len(user_input[i]), args.candidates-args.candidates_neg]) # int(args.candidates/2)
                negsamples_candidates_K2 = []
                u_id = 0
                for u in user_input[i]:
                    epoch_num += 1
                    # sample unobserved candidates in K1
                    l = 0
                    for k in negsamples_candidates_K1[u_id]:
                        if k in dataset_cur.trainDictSet[u] or (u in dataset_cur.trainNegDictArray and k in dataset_cur.trainNegDict[u]):
                            negsamples_candidates_K1[u_id, l] = random.randint(0, num_items - 1)
                            while (negsamples_candidates_K1[u_id, l] in dataset_cur.trainDictSet[u]
                                   or (u in dataset_cur.trainNegDictArray and negsamples_candidates_K1[u_id, l] in dataset_cur.trainNegDict[u])):
                                negsamples_candidates_K1[u_id, l] = random.randint(0, num_items - 1)
                        l += 1
                    # sample displayed but non-clicked candidates in K2
                    if args.candidates_neg > 0:
                        if u in dataset_cur.trainNegDictArray:
                            negsamples_candidates_K2.append(np.random.choice(dataset_cur.trainNegDictArray[u], args.candidates_neg)) # int(args.candidates/2)
                        else: # no displayed interactions
                            tmp = np.random.choice(num_items, args.candidates_neg) # int(args.candidates / 2)
                            for tt in range(len(tmp)):
                                if tmp[tt] in dataset_cur.trainDictSet[u]:
                                    tmp[tt] = random.randint(0, num_items - 1)
                                    while tmp[tt] in dataset_cur.trainDictSet[u]:
                                        tmp[tt] = random.randint(0, num_items - 1)
                            negsamples_candidates_K2.append(tmp)
                    u_id += 1
                if args.candidates_neg > 0:
                    negsamples_candidates_K2 = np.array(negsamples_candidates_K2)
                    negsamples_candidates_K = np.concatenate([negsamples_candidates_K1,negsamples_candidates_K2], axis=1)
                else:
                    negsamples_candidates_K = negsamples_candidates_K1
        # time
        train1 = time() * 1000

        # Step 2: GEN sample negative items
        if not args.reduced:
            feed_dict = {gen.user_input: np.reshape(user_input[i], [-1, 1]),
                         gen.i_pos: user_i_pos,
                         gen.num_neg: args.num_neg} # currently num_neg can only be 1!!!
            negsamples = sess.run(gen.negsamples, feed_dict)
        else:
            feed_dict = {gen.user_input: np.reshape(user_input[i], [-1, 1]),
                         gen.candidates_neg: negsamples_candidates_K,
                         gen.num_neg: args.num_neg}
            negsamples_id = np.reshape(sess.run(gen.negsamples, feed_dict), [1,-1]) # 1*n
            negsamples = np.reshape(negsamples_candidates_K[range(len(user_input[i])), negsamples_id], [-1,1]) # n*1
        # time
        train2 = time() * 1000

        # Step 3: Get reward from DIS, Update DIS
        if args.model == "KBGAN":
            feed_dict = {dis.user_input: np.reshape(user_input[i], [-1, 1]),
                         dis.item_input: np.reshape(item_input[i], [-1, 1]),
                         dis.item_input_neg: negsamples}
            _, loss_dis, reward = sess.run([dis.optimizer, dis.loss, dis.reward], feed_dict)
            reward_epoch += np.sum(reward)
            loss_dis_average += loss_dis
            # time
            train3 = time() * 1000

            # Reward NegID -- whether generated negs are displayed but non-clicked instances (Not used in KBGAN)
            u_id = 0
            reward_realNeg = np.zeros([len(user_input[i]), 1], dtype=np.float32)
            for u in user_input[i]:
                if u in dataset_cur.trainNegDict and negsamples[u_id][0] in dataset_cur.trainNegDict[u]:
                    reward_realNeg[u_id] = 1.0
                u_id += 1
        elif args.model == "RNS":
            feed_dict = {dis.user_input: np.reshape(user_input[i], [-1, 1]),
                         dis.item_input: np.reshape(item_input[i], [-1, 1]),
                         dis.item_input_neg: negsamples}
            _, loss_dis, reward = sess.run([dis.optimizer, dis.loss, dis.reward], feed_dict)
            reward_epoch += np.sum(reward)
            loss_dis_average += loss_dis
            # time
            train3 = time() * 1000

            # Reward NegFeature -- whether generated negs are close to displayed but non-clicked instances
            u_id = 0
            data_realNeg = np.zeros([len(user_input[i]), 1], dtype=np.float32)
            for u in user_input[i]: # sample a minibatch of displayed items
                if u in dataset_cur.trainNegDict:
                    data_realNeg[u_id,0] = np.random.choice(dataset_cur.trainNegDictArray[u])
                else:
                    data_realNeg[u_id,0] = random.randint(0, dataset_cur.num_items - 1)
                    while u in dataset_cur.trainDictSet and data_realNeg[u_id, 0] in dataset_cur.trainDictSet[u]:
                        data_realNeg[u_id,0] = random.randint(0, dataset_cur.num_items - 1)
                u_id += 1
            # Reward NegID -- whether generated negs are displayed but non-clicked instances
            u_id = 0
            reward_realNeg_id = np.zeros([len(user_input[i]), 1], dtype=np.float32)
            for u in user_input[i]:
                if u in dataset_cur.trainNegDict and negsamples[u_id][0] in dataset_cur.trainNegDict[u]:
                    reward_realNeg_id[u_id] = 1.0
                    cnt_realNeg += 1
                u_id += 1

            feed_dict = {dis.user_input: np.reshape(user_input[i], [-1, 1]),
                         dis.item_input: data_realNeg,
                         dis.item_input_neg: negsamples} #
            [reward_mmd,reward_mmd_array] = sess.run([dis.reward_mmd,dis.reward_mmd_array], feed_dict)
            # RNS combine two reward (about generating real negative) by beta
            reward_realNeg = reward_realNeg_id * (1 - args.beta) + reward_mmd_array * args.beta
            reward_realNegId_epoch += np.sum(reward_realNeg_id)
            reward_realNegFeature_epoch += reward_mmd * np.shape(reward_realNeg)[0]

        # Step 4: Update GEN
        if not args.neg_samples_only:
            # KBGAN/RNS No Reduced Sampling space (Do not work)
            if not args.reduced:
                feed_dict = {gen.user_input: np.reshape(user_input[i], [-1, 1]),
                             gen.item_input: negsamples,
                             gen.i_pos: user_i_pos,
                             gen.reward: reward-reward_average,
                             gen.reward_realNeg: reward_realNeg-reward_realNeg_average}
                _, loss_gen = sess.run([gen.optimizer, gen.loss], feed_dict)
            # KBGAN/RNS with Reduced Sampling space (Recommended)
            else:
                # KBGAN (with args.alpha=0) / RNS
                if not args.NoDNSLoss:
                    feed_dict = {gen.user_input: np.reshape(user_input[i], [-1, 1]),
                                 gen.item_input: negsamples,
                                 gen.candidates_neg: negsamples_candidates_K,
                                 gen.item_id_input: np.reshape(negsamples_id, [-1,1]),
                                 gen.reward: reward - reward_average,
                                 gen.reward_realNeg: reward_realNeg - reward_realNeg_average}
                # variant of RNS (No DNS Loss)
                else:
                    feed_dict = {gen.user_input: np.reshape(user_input[i], [-1, 1]),
                                 gen.item_input: negsamples,
                                 gen.candidates_neg: negsamples_candidates_K,
                                 gen.item_id_input: np.reshape(negsamples_id, [-1, 1]),
                                 gen.reward: reward_realNeg - reward_realNeg_average,
                                 gen.reward_realNeg: np.zeros([len(user_input[i]), 1], dtype=np.float32)}
                _, loss_gen = sess.run([gen.optimizer, gen.loss], feed_dict)

            reward_realNeg_epoch += np.sum(reward_realNeg)
            loss_gen_average += loss_gen
        # time
        train4 = time() * 1000

    end0 = time()

    reward_realNeg_average = reward_realNeg_epoch / (0.+epoch_num)
    reward_average = reward_epoch / (0.+epoch_num)
    loss_dis_average /= epoch_num
    loss_gen_average /= epoch_num

    if args.model != "RNS":
        print ("Epoch Size: %d  Time: %.1fs Reward-Neg: %.1f average Reward/Reward-Neg: %.4f/%.4f cnt-realNeg: %d" % (
        epoch_num, end0 - begin0, reward_realNeg_epoch, reward_average, reward_realNeg_average, cnt_realNeg))
        logging.info("Epoch Size %d Reward_Neg %d average Reward/Reward-Neg: %.4f/%.4f cnt-realNeg: %d" % (epoch_num, reward_realNeg_epoch, reward_average, reward_realNeg_average, cnt_realNeg))
    else:
        print ("Epoch Size: %d  Time: %.1fs Reward-Neg: %.1f average Reward/NegId/NegFeature: %.4f/%.4f/%.4f cnt-realNeg: %d" % (
            epoch_num, end0 - begin0, reward_realNeg_epoch, reward_average, reward_realNegId_epoch/epoch_num, reward_realNegFeature_epoch/epoch_num, cnt_realNeg))
        logging.info("Epoch Size: %d  Reward-Neg: %.1f average Reward/NegId/NegFeature: %.4f/%.4f/%.4f cnt-realNeg: %d" % (
        epoch_num, reward_realNeg_epoch, reward_average, reward_realNegId_epoch/epoch_num, reward_realNegFeature_epoch/epoch_num, cnt_realNeg))

    return loss_dis_average, loss_gen_average, reward_average, reward_realNeg_average

# used for single task learning
def pretraining_batch(model, sess, batches, args):

    num_batch = len(batches[1])
    epoch_num = 0
    loss_train = 0.0
    loss_train_all = 0.0

    user_input, item_input, item_input_neg = batches
    for i in range(len(user_input)):
        epoch_num += len(user_input[i])
        if args.pretrain_dis: # BPR Loss
            feed_dict = {model.user_input: np.reshape(user_input[i], [-1, 1]),
                         model.item_input: np.reshape(item_input[i], [-1, 1]),
                         model.item_input_neg: np.reshape(item_input_neg[i], [-1, 1])}
        else:
            print 'ERROR! Pls specify loss!'
            return 0

        _, loss_train = sess.run([model.optimizer, model.loss], feed_dict)
        loss_train_all += loss_train

    return loss_train_all / epoch_num # num_batch

def init_logging_and_result(args):
    global filename
    global param_filename # saving param, filename cannot contain []
    global model_filename

    path_log = Log_dir_name
    if not os.path.exists(path_log):
        os.makedirs(path_log)

    # define factors
    dis_model = args.dis_model
    if dis_model == "MLP":
        dis_model += "_L"+str(args.layer_num)
    gen_model = args.gen_model
    if gen_model == "MLP":
        gen_model += "_L" + str(args.layer_num)
    if args.pretrain_dis:
        F_model = 'pretrain-' + dis_model + '-dis'
    elif args.pretrain_gen:
        F_model = 'pretrain-' + gen_model + '-gen'
    else:
        if args.reduced:
            F_model = args.model + '-Red' + str(args.candidates) + "," + str(args.candidates_neg)
        elif args.model == "KBGAN" or args.model == "RNS":
            F_model = args.model + '-C' + str(args.c_entropy)
        else:
            F_model = args.model
        if args.use_pretrain_dis:
            F_model += '-dis-' + dis_model + '-p'
        else:
            F_model += '-dis-' + dis_model + '-s'
        if args.use_pretrain_gen:
            F_model += '-gen-' + gen_model + '-p'
        else:
            F_model += '-gen-' + gen_model + '-s'
    F_dataset = args.dataset
    F_embedding = args.embed_size
    if args.eval_mode == "topK":
        F_topK = args.topK
    else:
        F_topK = "List"+str(args.LRecList)
    F_num_neg = args.num_neg
    if args.dns:
        F_trail_id = 'pregen-dns-' + args.trial_id
    else:
        F_trail_id = args.trial_id
    if args.lr_g == -1:
        F_optimizer = args.optimizer + str(args.lr)
    else:
        F_optimizer = args.optimizer + str(args.lr) + "," + str(args.lr_g)
    F_lr = args.lr
    F_reg = args.regs

    if args.model == "DNS":
        F_trail_id = str(args.K_DNS)+"-"+F_trail_id
    elif args.model == "KBGAN":
        F_trail_id = "-alpha"+str(args.alpha)+"-temp" + str(args.temperature) + "-" + F_trail_id
    elif args.model == "RNS" and not args.NoDNSLoss and args.neg_samples_only:
        F_trail_id = "-alpha" + str(args.alpha) + "-temp" + str(args.temperature) + "-negsample" + str(
            args.neg_samples_ratio) + "-" + F_trail_id
    elif args.model == "RNS" and not args.NoDNSLoss:
        F_trail_id = "-alpha"+str(args.alpha)+"-beta"+str(args.beta)+"-temp"+str(args.temperature)+"-sigma"+args.sigma_range+"-"+F_trail_id
    elif args.model == "RNS" and args.NoDNSLoss:
        F_trail_id = "-alphaNoDNSLoss-beta"+str(args.beta)+"-temp"+str(args.temperature)+"-sigma"+args.sigma_range+"-"+F_trail_id

    if "zhihu" in F_dataset:
        F_run_mode = '-zhihu'

    filename = "log-%s-%s-dim%s-topK%s-numneg%s-%s-%s-lr%s%s%s" % (
        F_model, F_dataset, F_embedding, F_topK,
        F_num_neg, F_optimizer, F_reg, F_lr, F_run_mode, F_trail_id)

    # if already exec, return 0
    if not os.path.exists(Log_dir_name + '/' + filename):
        logging.basicConfig(filename=path_log + '/' + filename, level=logging.INFO)
        logging.info('Use Multiprocess to Evaluate: %s' % args.multiprocess)
    else:
        print(Log_dir_name + '/' + filename, 'already exists, skipping ...')
        exit(0)

    # param filename
    if args.save_param:
        param_filename = "dim%s-topK%s-lr%s-%s%s%s" % (F_embedding, F_topK, F_lr, '-'.join([str(s) for s in eval(F_reg)]), F_run_mode, F_trail_id)
        print(param_filename)
        if not os.path.exists(Param_dir_name):
            os.mkdir(Param_dir_name)
        if not os.path.exists(Param_dir_name + '/' + args.model):
            os.mkdir(Param_dir_name + '/' + args.model)
        if not os.path.exists(Param_dir_name + '/' + args.model + '/' + param_filename):
            os.mkdir(Param_dir_name + '/' + args.model + '/' + param_filename)

    # model filename
    if args.save_model:
        model_filename = "%s-dim%s-topK%s-numneg%s-%s-%s%s%s" % (
            F_model, F_embedding, F_topK, F_num_neg, F_optimizer, '-'.join([str(s) for s in eval(F_reg)]), F_run_mode, F_trail_id)
        print(model_filename)
        if not os.path.exists(Model_dir_name):
            os.mkdir(Model_dir_name)
        if not os.path.exists(Model_dir_name + '/' + args.model):
            os.mkdir(Model_dir_name + '/' + args.model)
        if not os.path.exists(Model_dir_name + '/' + args.model + '/Pretrain'):
            os.mkdir(Model_dir_name + '/' + args.model + '/Pretrain')
        if not os.path.exists(Model_dir_name + '/' + args.model + '/Pretrain/DIS'):
            os.mkdir(Model_dir_name + '/' + args.model + '/Pretrain/DIS')
        if not os.path.exists(Model_dir_name + '/' + args.model + '/Pretrain/GEN'):
            os.mkdir(Model_dir_name + '/' + args.model + '/Pretrain/GEN')
        if args.pretrain_dis:
            if not os.path.exists(Model_dir_name + '/' + args.model + '/Pretrain/DIS/' + model_filename):
                os.mkdir(Model_dir_name + '/' + args.model + '/Pretrain/DIS/' + model_filename)
        elif args.pretrain_gen:
            if not os.path.exists(Model_dir_name + '/' + args.model + '/Pretrain/GEN/' + model_filename):
                os.mkdir(Model_dir_name + '/' + args.model + '/Pretrain/GEN/' + model_filename)
        else:
            if not os.path.exists(Model_dir_name + '/' + args.model + '/' + model_filename):
                os.mkdir(Model_dir_name + '/' + args.model + '/' + model_filename)

def save_results(args):
    path_result = Result_dir_name # it is better to use relative path

    if not os.path.exists(path_result):
        os.makedirs(path_result)

    with open(path_result + '/' + filename, 'w') as output:
        for i in range(args.epochs):
            output.write('%.4f,%.4f,%.4f,%.4f\n' % (loss_dis_list[i], loss_gen_list[i], hr_list[i], ndcg_list[i]))



if __name__ == '__main__':
    args = parse_args()

    if args.eval_mode == "topK":
        Log_dir_name = 'Log'
        Result_dir_name = 'Result'
        Param_dir_name = 'Param'
        Model_dir_name = 'Model'
    elif args.eval_mode == "list" or args.eval_mode == "report":
        Log_dir_name = 'Log-RecList'
        Result_dir_name = 'Result-RecList'
        Param_dir_name = 'Param-RecList'
        Model_dir_name = 'Model-RecList'

    assert (args.pretrain_dis and args.pretrain_gen) is False

    dataset_cur = None

    filename = None
    eval_queue = JoinableQueue()
    job_num = Semaphore(0)
    job_lock = Lock()
#
    loss_list = range(args.epochs)
    loss_dis_list = range(args.epochs)
    loss_gen_list = range(args.epochs)
    hr_list = range(args.epochs)
    ndcg_list = range(args.epochs)

    # initialize logging and configuration
    print('------ %s ------' % (args.process_name))
    setproctitle.setproctitle(args.process_name)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.eval_mode != "report":
        init_logging_and_result(args)


    # load data
    print('--- Loading data and data generation start ---')

    num_users, num_items = 0, 0
    dir_path = ''


    if args.dataset == 'zhihu_click_data':
        print('Load zhihu data')
        num_users, num_items = 16015, 45782
        data_gen_begin = time()
        if args.eval_mode == 'topK':
            dir_path = 'data/zhihu/'
            dataset_cur = Dataset(dir_path + 'data.zhihu.display.train', dir_path + 'data.zhihu.display.validation',
                                  num_users, num_items, args.candidates)  # data
        elif args.eval_mode == 'list' or args.eval_mode == "report":
            dir_path = 'data/zhihu/'
            dataset_cur = Dataset(dir_path + 'data.zhihu.display.train', dir_path + 'data.zhihu.display.validation',
                                  num_users, num_items, args.candidates,
                                  report_path=dir_path + 'data.zhihu.display.test', evalRecList = True, LRecList = args.LRecList, write = args.write_list)  # data

    print('data generation finished: cost [%.1f s]' % (time() - data_gen_begin))

    if args.eval_mode == "report":
        # report model (tuned on validation) metrics on test
        with open("report.model."+args.dataset.split("_")[0]) as f:
            # Dataset,Model,Module,path
            for line in f:
                Dataset, Model, Module, path = line.strip().split("|")
                if Model == 'ItemPop':
                    print('start building ItemPop')
                    tf.reset_default_graph
                    graph = tf.Graph()
                    dis = ItemPop(dataset_cur.num_users, dataset_cur.num_items,
                                  dataset_cur.items_pop_score, args, reclist_len=args.LRecList)
                    dis.build_graph()
                else:
                    if Model == 'RNS' or Model == 'KBGAN' or Model == 'DNS':
                        print('start building RNS/KBGAN/DNS')
                        param_dis = cPickle.load(open(Model_dir_name + "/" + Model + "/" + path + "/" + "dis_models.pkl"))

                    elif Model == 'Pretrain-DIS':
                        print('start building Pretrain')
                        param_dis = cPickle.load(open(Model_dir_name + "/RNS/Pretrain/DIS/" + path + "/" + "pretrain_model_dis.pkl"))

                    args.dis_model = Module
                    tf.reset_default_graph
                    graph = tf.Graph()
                    dis = DIS(dataset_cur.num_users, dataset_cur.num_items, args,
                              use_pretrain=True, param=param_dis, reclist_len=args.LRecList)
                    dis.build_graph(mode=args.reg_mode)
                reportModelMetrics(dis=dis, args=args)
    else:
        # training with topK/list mode
        if args.model == 'KBGAN' or args.model == 'RNS':
            print('start building KBGAN/RNS')

            param_gen = None
            param_dis = None
            if args.use_pretrain_gen:
                param_gen = cPickle.load(open(Model_dir_name + "/" + args.gen_file))
            if args.use_pretrain_dis:
                param_dis = cPickle.load(open(Model_dir_name + "/" + args.dis_file))
            graph = tf.Graph()
            gen = GEN(dataset_cur.num_users, dataset_cur.num_items, args, use_pretrain=args.use_pretrain_gen,
                      param=param_gen, reclist_len=args.LRecList)
            gen.build_graph(mode=args.reg_mode)
            dis = DIS(dataset_cur.num_users, dataset_cur.num_items, args, use_pretrain=args.use_pretrain_dis,
                      param=param_dis, reclist_len=args.LRecList)
            dis.build_graph(mode=args.reg_mode)

            if args.pretrain_dis:
                pretraining(model=dis, train="dis", args=args)
            # gen and dis use the same parameters for pre-training, pretrain_gen is inactive
            # elif args.pretrain_gen:
            #     pretraining(model=gen, train="gen", args=args)
            else:
                if args.num_neg > 1:
                    print "currently num_neg can only be 1!!!"
                else:
                    training(dis=dis,gen=gen,args=args)

        elif args.model == 'DNS':
            print('start building DNS')

            param_gen = None
            param_dis = None
            if args.use_pretrain_dis:
                param_dis = cPickle.load(open(Model_dir_name + "/" + args.dis_file))
            graph = tf.Graph()
            gen = DNS(dataset_cur.num_users, dataset_cur.num_items, args)
            gen.build_graph()
            dis = DIS(dataset_cur.num_users, dataset_cur.num_items, args, use_pretrain=args.use_pretrain_dis,
                      param=param_dis, reclist_len=args.LRecList)
            dis.build_graph()

            training(dis=dis, gen=gen, args=args)

        elif args.model == 'ItemPop':
            print('ItemPop model')
            itempop = ItemPop(dataset_cur.num_users, dataset_cur.num_items, dataset_cur.items_pop_score, args, reclist_len=args.LRecList)
            itempop.build_graph()
            training(dis=itempop,gen=itempop, args=args)

        if not args.pretrain_gen and not args.pretrain_dis:
            save_results(args)

