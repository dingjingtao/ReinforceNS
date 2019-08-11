from __future__ import division

import os

import tensorflow as tf
import cPickle
import numpy as np
import scipy.io as sio

from baseline_model import GMF, MLP

class DIS():
    def __init__(self, num_users, num_items, args, use_pretrain=False, param=None, reclist_len=1):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        self.learning_rate = args.lr
        self.opt = args.optimizer
        regs = eval(args.regs)
        self.lambda_bilinear = regs[0]
        self.mode = not args.pretrain_dis
        self.Lreclist = reclist_len
        self.candidates = args.candidates
        self.sigma_range = eval(args.sigma_range)

        if args.dis_model == 'GMF':
            self.model = GMF(self.num_users, self.num_items, self.embedding_size, regs, use_pretrain=use_pretrain, param=param)
        elif args.dis_model == 'MLP':
            self.model = MLP(self.num_users, self.num_items, self.embedding_size, args.layer_num, regs, use_pretrain=use_pretrain,
                             param=param)
        else:
            raise NameError("null model")

    def _create_placeholders(self):
        with tf.name_scope("DIS"):
            with tf.name_scope("input_data"):
                self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
                self.item_input = tf.placeholder(tf.int32, shape=[None, 1], name="item_input")
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")
                self.candidates_neg = tf.placeholder(tf.int32, shape=[None, self.candidates], name="candidates_neg")
                self.candidates_reclist = tf.placeholder(tf.int32, shape=[None, self.Lreclist], name="candidates_reclist")
                ##
                self.model._create_placeholders([self.user_input,self.item_input,self.candidates_neg,self.candidates_reclist])
                # self.model._create_placeholders([self.user_input,self.item_input])

    def _create_variables(self):
        with tf.name_scope("DIS"):
            self.model._create_variables()

    def _create_loss(self, mode = True):
        with tf.name_scope("DIS"):
            self.model._create_loss()
            with tf.name_scope("loss"):
                self.output = self.model._create_inference(self.item_input)
                self.output_neg = self.model._create_inference(self.item_input_neg)

                if self.mode: # train: BPR-loss
                    self.result = self.output - self.output_neg
                    if self.lambda_bilinear > 0:
                        if mode:
                            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) + \
                                        self.model.reg_loss_em(self.user_input, type="user") + \
                                        self.model.reg_loss_em(self.item_input, type="item") + \
                                        self.model.reg_loss_em(self.item_input_neg, type="item") +\
                                        self.model.loss_reg_w
                        else:
                            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) + \
                                        self.model.loss_reg
                    else:
                        self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result)))
                else: # pre-train: BPR-loss
                    self.result = self.output - self.output_neg
                    if self.lambda_bilinear > 0:
                        if mode:
                            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) + \
                                        self.model.reg_loss_em(self.user_input, type="user") + \
                                        self.model.reg_loss_em(self.item_input, type="item") + \
                                        self.model.reg_loss_em(self.item_input_neg, type="item") + \
                                        self.model.loss_reg_w
                        else:
                            self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result))) + \
                                        self.model.loss_reg
                    else:
                        self.loss = tf.reduce_sum(tf.log(1 + tf.exp(-self.result)))

    def _create_allrating(self):
        with tf.name_scope("DIS"):
            with tf.name_scope('all_rating'):
                self.all_rating = self.model.all_logits
            with tf.name_scope('list_rating'):
                self.list_rating = self.model.list_logits
                _, list_indices = tf.nn.top_k(self.model.list_logits, k=self.Lreclist)
                _, self.list_order = tf.nn.top_k(-list_indices, k=self.Lreclist)
            with tf.name_scope('sampled_rating'):
                self.sampled_rating = self.model.sampled_logits

    def _create_optimizer(self):
        with tf.name_scope("DIS"):
            with tf.name_scope('optimizer'):
                if self.opt == 'Adam':
                    self.optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate).minimize(self.loss, var_list=self.model.model_params)
                elif self.opt == 'Adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(
                        learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss, var_list=self.model.model_params)
                elif self.opt == 'GradientDescent':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.model.model_params)

    def _compute_MMD(self, item_input_real, item_input_fake):
        dist_x = self.model._create_feature(item_input_real) # n*K' feature_real
        dist_y = self.model._create_feature(item_input_fake) # n*K' feature_fake

        x_sq = tf.expand_dims(tf.reduce_sum(dist_x ** 2, axis=1), 1)  # n*1
        y_sq = tf.expand_dims(tf.reduce_sum(dist_y ** 2, axis=1), 1)  # n*1
        dist_x_T = tf.transpose(dist_x)
        dist_y_T = tf.transpose(dist_y)
        x_sq_T = tf.transpose(x_sq)
        y_sq_T = tf.transpose(y_sq)

        tempxx = -2 * tf.matmul(dist_x, dist_x_T) + x_sq + x_sq_T  # (xi -xj)**2 Size: n*n
        tempxy = -2 * tf.matmul(dist_x, dist_y_T) + x_sq + y_sq_T  # (xi -yj)**2 Size: n*n
        tempyy = -2 * tf.matmul(dist_y, dist_y_T) + y_sq + y_sq_T  # (yi -yj)**2 Size: n*n

        kxx, kxy, kyy = 0.0, 0.0, 0.0
        kxy_array = tf.zeros([tf.shape(item_input_fake)[0]], dtype=tf.float32)
        kyy_array = tf.zeros([tf.shape(item_input_fake)[0]], dtype=tf.float32)
        for sigma in self.sigma_range: # sigma1, sigma2, ...
            # kxx, kxy, kyy = 0.0, 0.0, 0.0
            kxx += tf.reduce_mean(tf.exp(-tempxx / 2 / (sigma ** 2)))
            kxy += tf.reduce_mean(tf.exp(-tempxy / 2 / (sigma ** 2)))
            kyy += tf.reduce_mean(tf.exp(-tempyy / 2 / (sigma ** 2)))
            kxy_array += tf.reduce_mean(tf.exp(-tempxy / 2 / (sigma ** 2)), axis=0)
            kyy_array += tf.reduce_mean(tf.exp(-tempyy / 2 / (sigma ** 2)), axis=0)
        mmd_array = tf.reshape(2 * kyy_array - 2 * kxy_array, [-1,1])

        return -tf.sqrt(kxx + kyy - 2 * kxy), -mmd_array

    def _create_reward(self):
        with tf.name_scope("DIS"):
            self.reward = -tf.sigmoid(-self.model._create_inference(self.item_input_neg))

            [self.reward_mmd,self.reward_mmd_array] = self._compute_MMD(self.item_input, self.item_input_neg)
            self.f1 = self.model._create_feature(self.item_input)
            self.f2 = self.model._create_feature(self.item_input_neg)

    def build_graph(self, mode = True):
        graph = tf.get_default_graph()
        with graph.as_default():
            self._create_placeholders()
            self._create_variables()
            self._create_loss(mode)
            self._create_allrating()
            self._create_optimizer()
            self._create_reward()

    def save_model(self, sess, filename):
        self.model.save_model(sess, filename)