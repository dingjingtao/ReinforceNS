from __future__ import division

import os

import tensorflow as tf
import cPickle
import numpy as np

from baseline_model import GMF, MLP

class GEN():
    def __init__(self, num_users, num_items, args, use_pretrain=False, param=None, reclist_len=1):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = args.embed_size
        if args.lr_g == -1:
            self.learning_rate = args.lr
        else:
            self.learning_rate = args.lr_g
        self.opt = args.optimizer
        regs = eval(args.regs)
        self.lambda_bilinear = regs[1]
        self.alpha = args.alpha
        self.mode = not args.pretrain_gen
        self.temperature = args.temperature
        self.candidates = args.candidates
        self.reduced = args.reduced
        self.Lreclist = reclist_len
        self.c_entropy = args.c_entropy


        if args.gen_model == 'GMF':
            self.model = GMF(self.num_users, self.num_items, self.embedding_size, regs, use_pretrain=use_pretrain, param=param)
        elif args.gen_model == 'MLP':
            self.model = MLP(self.num_users, self.num_items, self.embedding_size, args.layer_num, regs, use_pretrain=use_pretrain,
                             param=param)
        else:
            raise NameError("null model")

    def _create_placeholders(self):
        with tf.name_scope("GEN"):
            with tf.name_scope("input_data"):
                self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
                self.item_input = tf.placeholder(tf.int32, shape=[None, 1], name="item_input")
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")
                self.labels = tf.placeholder(tf.int32, shape=[None, 1], name="labels")  # (b,1)
                self.reward_realNeg = tf.placeholder(tf.float32, shape=[None, 1], name="reward_realNeg")
                self.reward = tf.placeholder(tf.float32, shape=[None, 1], name="reward")
                self.i_pos = tf.placeholder(tf.int64, shape=[None, 2], name="i_pos") # [ [u1,i1], [u1,i2], [u2,i3], ... ]
                self.num_neg = tf.placeholder(tf.int32, shape=None, name="num_neg")
                self.candidates_neg = tf.placeholder(tf.int32, shape=[None, self.candidates], name="candidates_neg")
                self.item_id_input = tf.placeholder(tf.int32, shape=[None, 1], name="item_id_input")
                self.candidates_reclist = tf.placeholder(tf.int32, shape=[None, self.Lreclist], name="candidates_reclist")
                ##
                self.model._create_placeholders([self.user_input, self.item_input, self.candidates_neg, self.candidates_reclist])
                # self.model._create_placeholders([self.user_input,self.item_input])

    def _create_variables(self):
        with tf.name_scope("GEN"):
            self.model._create_variables()

    def _gen_negsample(self):
        self.model._create_loss()
        user_i_pos = tf.SparseTensor(indices=self.i_pos, values=tf.ones([tf.shape(self.i_pos)[0]],dtype=tf.float32),
                                     dense_shape = [tf.shape(self.user_input, out_type=tf.int64)[0],self.num_items])
        # all_prob = tf.exp(self.model.all_logits)
        # all_prob_masked = tf.sparse_add(all_prob, user_i_pos*(-1)*all_prob)
        if not self.reduced:
            self.all_logits_masked = tf.sparse_add(self.model.all_logits / self.temperature, user_i_pos*(-np.inf))
            # self.prob_negsample = all_prob_masked/(tf.reduce_sum(all_prob_masked,axis=1)[:,None]) # n * M i_pos -> prob=0
        else:
            # for reduced sampling
            self.all_logits_masked = self.model.sampled_logits / self.temperature

        # self.negsamples = tf.reshape(tf.multinomial(self.all_logits_masked, self.num_neg, output_dtype=tf.int32), [-1, 1])
        self.negsamples = tf.reshape(tf.multinomial(self.all_logits_masked, self.num_neg), [-1, 1])

        # self.negprobs = tf.gather_nd(prob_negsample, tf.concat([tf.range(0,tf.shape(self.user_input)[0])[:,None], self.negsamples], axis=1))[:,None]


    def _create_loss(self, mode = True):
        with tf.name_scope("GEN"):
            with tf.name_scope("loss"):
                self.output = self.model._create_inference(self.item_input)

                if self.mode: # train: gen_loss + realNeg_loss
                    self.prob_negsample = tf.nn.softmax(self.all_logits_masked)
                    if not self.reduced:
                        self.i_prob = tf.gather_nd(self.prob_negsample, tf.concat(
                            [tf.range(0, tf.shape(self.user_input)[0])[:, None], self.item_input], axis=1), name="i_prob")[:, None]
                        self.entropy_gen = - tf.reduce_sum(
                            self.prob_negsample * tf.log(tf.clip_by_value(self.prob_negsample, 1e-10, 1.0)), axis=1)
                        self.loss_entropy = tf.reduce_sum(tf.minimum(0.0, np.log(self.c_entropy) - self.entropy_gen))
                    else:
                        self.i_prob = tf.gather_nd(self.prob_negsample, tf.concat(
                            [tf.range(0, tf.shape(self.user_input)[0])[:, None], self.item_id_input], axis=1),
                                                   name="i_prob")[:, None]
                        self.loss_entropy = 0.0
                    if self.lambda_bilinear > 0:
                        if mode:
                            self.loss = - tf.reduce_sum(tf.log(self.i_prob) * (self.reward + self.alpha * self.reward_realNeg)) + \
                                        self.model.reg_loss_em(self.user_input, type="user") + \
                                        self.model.reg_loss_em(self.item_input, type="item") + \
                                        self.model.loss_reg_w + \
                                        self.loss_entropy
                        else:
                            self.loss = - tf.reduce_sum(tf.log(self.i_prob) * (self.reward + self.alpha * self.reward_realNeg)) \
                                        + self.model.loss_reg + \
                                        self.loss_entropy
                    else:
                        self.loss_list = - tf.log(self.i_prob) * (self.reward)  # debug
                        self.loss = - tf.reduce_sum(
                            tf.log(self.i_prob) * (self.reward + self.alpha * self.reward_realNeg)) + \
                                        self.loss_entropy

                else: # pre-train: log-loss
                    print ("No pretrain-gen mode!!! -> exit")
                    exit(0)

    def _create_allrating(self):
        with tf.name_scope("GEN"):
            with tf.name_scope('all_rating'):
                self.all_rating = self.model.all_logits
            with tf.name_scope('list_rating'):
                self.list_rating = self.model.list_logits
                _, list_indices = tf.nn.top_k(self.model.list_logits, k=self.Lreclist)
                _, self.list_order = tf.nn.top_k(-list_indices, k=self.Lreclist)
            with tf.name_scope('sampled_rating'):
                self.sampled_rating = self.model.sampled_logits

    def _create_optimizer(self):
        with tf.name_scope("GEN"):
            with tf.name_scope('optimizer'):
                if self.opt == 'Adam':
                    self.optimizer = tf.train.AdamOptimizer(
                        learning_rate=self.learning_rate).minimize(self.loss, var_list=self.model.model_params)
                elif self.opt == 'Adagrad':
                    self.optimizer = tf.train.AdagradOptimizer(
                        learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss, var_list=self.model.model_params)
                elif self.opt == 'GradientDescent':
                    self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss, var_list=self.model.model_params)

    def build_graph(self, mode=True):
        graph = tf.get_default_graph()
        with graph.as_default():
            self._create_placeholders()
            self._create_variables()
            self._gen_negsample()
            self._create_loss(mode)
            self._create_allrating()
            self._create_optimizer()

    def save_model(self, sess, filename):
        self.model.save_model(sess, filename)

class DNS():
    def __init__(self, num_users, num_items, args):
        self.num_items = num_items
        self.num_users = num_users
        self.K_DNS = args.K_DNS

    def _create_placeholders(self):
        with tf.name_scope("GEN"):
            with tf.name_scope("input_data"):
                self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
                self.item_input_neg = tf.placeholder(tf.int32, shape=[None, 1], name="item_input_neg")
                self.i_pos = tf.placeholder(tf.int64, shape=[None, 2], name="i_pos")

    def _gen_negsample(self):
        user_i_pos = tf.SparseTensor(indices=self.i_pos, values=tf.ones([tf.shape(self.i_pos)[0]], dtype=tf.float32),
                                     dense_shape=[tf.shape(self.user_input, out_type=tf.int64)[0], self.num_items])
        self.all_logits_masked = tf.sparse_add(tf.ones([tf.shape(self.user_input)[0], self.num_items], dtype=tf.float32), user_i_pos * (-np.inf))
        # self.negsamples = tf.multinomial(self.all_logits_masked, self.K_DNS, output_dtype=tf.int32)
        self.negsamples = tf.multinomial(self.all_logits_masked, self.K_DNS)

    def build_graph(self):
        graph = tf.get_default_graph()
        with graph.as_default():
            self._create_placeholders()
            self._gen_negsample()


