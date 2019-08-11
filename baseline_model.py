from __future__ import absolute_import
from __future__ import division

import os

import tensorflow as tf
import cPickle
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class MLP:
    def __init__(self, num_users, num_items, embed_size, layer_num, regs, use_pretrain=False, param=None):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = embed_size
        self.layer_num = layer_num
        self.param = param
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.use_pretrain = use_pretrain

    def _create_placeholders(self, placeholders):
        with tf.name_scope("input_data"):
            self.user_input = placeholders[0]
            self.item_input = placeholders[1]
            self.candidates_neg = placeholders[2]
            self.candidates_reclist = placeholders[3]

    def _create_variables(self):
        with tf.name_scope("embedding"):
            if self.use_pretrain:
                self.embedding_P = tf.Variable(self.param[0], name='embedding_P')
                self.embedding_Q = tf.Variable(self.param[1], name='embedding_Q')
                self.h = tf.Variable(self.param[2], name='h')
            else:
                self.embedding_P = tf.Variable(
                    tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                    name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
                self.embedding_Q = tf.Variable(
                    tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                    name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)
                self.h = tf.Variable(
                    tf.random_uniform([2 * int(self.embedding_size / (2 ** self.layer_num)), 1],
                                      minval=-tf.sqrt(3 / (2 * int(self.embedding_size / (2 ** self.layer_num)))),
                                      maxval=tf.sqrt(3 / (2 * int(self.embedding_size / (2 ** self.layer_num))))),
                    name='h')
        with tf.name_scope("FC"):
            if self.use_pretrain:
                if self.layer_num == 0:
                    self.model_params = [self.embedding_P, self.embedding_Q, self.h]
                    pass
                elif self.layer_num == 1:
                    self.W_FC = tf.Variable(self.param[3][0], name='W_FC')
                    self.b_FC = tf.Variable(self.param[3][1], name='b_FC')
                    self.model_params = [self.embedding_P, self.embedding_Q, self.h, self.W_FC, self.b_FC]
                else:
                    self.W_FC = []
                    self.b_FC = []
                    for i in range(self.layer_num):
                        self.W_FC.append(tf.Variable(self.param[3+i][0], name='W_FC_%d' % i))
                        self.b_FC.append(tf.Variable(self.param[3+i][1], name='b_FC_%d' % i))
                    self.model_params = [self.embedding_P, self.embedding_Q, self.h, self.W_FC, self.b_FC]
            else:
                if self.layer_num == 0:
                    self.model_params = [self.embedding_P, self.embedding_Q, self.h]
                    pass
                elif self.layer_num == 1:
                    # Xavier's Uniform Init (or He's ~ for deep layer of ReLU)
                    self.W_FC = tf.Variable(
                        tf.random_uniform(shape=[2 * self.embedding_size, int(2 * self.embedding_size / 2)],
                                          minval=-tf.sqrt(6 / (int(2 * self.embedding_size / 2))),
                                          maxval=tf.sqrt(6 / (int(2 * self.embedding_size / 2)))), name='W_FC')
                    self.b_FC = tf.Variable(tf.zeros([1, int(2 * self.embedding_size / 2)]), dtype=tf.float32, name='b_FC')
                    self.model_params = [self.embedding_P, self.embedding_Q, self.h, self.W_FC, self.b_FC]
                else:
                    self.W_FC = []
                    self.b_FC = []
                    for i in range(self.layer_num):
                        input_size = int(2 * self.embedding_size / (2 ** i))
                        output_size = int(2 * self.embedding_size / (2 ** (i + 1)))
                        self.W_FC.append(tf.Variable(tf.random_uniform(shape=[input_size, output_size],
                                                                       minval=-tf.sqrt(6 / (input_size+output_size)),
                                                                       maxval=tf.sqrt(6 / (input_size+output_size))), name='W_FC_%d' % i))
                        self.b_FC.append(tf.Variable(tf.zeros([1, output_size]), dtype=tf.float32, name='b_FC_%d' % i))
                    self.model_params = [self.embedding_P, self.embedding_Q, self.h, self.W_FC, self.b_FC]


    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  # (b, embedding_size)
            # self.concat_vec = tf.concat([self.embedding_p, self.embedding_q, self.embedding_p * self.embedding_q], 1,
            #                             name='concat_vec')
            concat_vec = tf.concat([embedding_p, embedding_q], 1, name='concat_vec')

            if self.layer_num == 0:
                # no sigmoid for BPR loss
                return tf.matmul(concat_vec, self.h, name='output') # (b, embedding_size) * (embedding_size, 1)

            elif self.layer_num == 1:
                fc = tf.nn.relu(tf.matmul(concat_vec, self.W_FC) + self.b_FC)
                # no sigmoid for BPR loss
                return tf.matmul(fc, self.h, name='output')

            else:
                fc = []
                for i in range(self.layer_num):
                    if i == 0:
                        fc.append(tf.nn.relu(tf.matmul(concat_vec, self.W_FC[i]) + self.b_FC[i]))
                    else:
                        fc.append(tf.nn.relu(tf.matmul(fc[i - 1], self.W_FC[i]) + self.b_FC[i]))
                # no sigmoid for BPR loss
                return tf.matmul(fc[i], self.h, name='output')

    def _create_feature(self, item_input):
        with tf.name_scope("feature"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)
            concat_vec = tf.concat([embedding_p, embedding_q], 1)

            if self.layer_num == 0:
                return concat_vec

            elif self.layer_num == 1:
                fc = tf.nn.relu(tf.matmul(concat_vec, self.W_FC) + self.b_FC)
                return fc

            else:
                fc = []
                for i in range(self.layer_num):
                    if i == 0:
                        fc.append(tf.nn.relu(tf.matmul(concat_vec, self.W_FC[i]) + self.b_FC[i]))
                    else:
                        fc.append(tf.nn.relu(tf.matmul(fc[i - 1], self.W_FC[i]) + self.b_FC[i]))
                return fc[i]

    # for computing n*M ratings in batch mode (batch size: n) --> update to einsum/tensordot (in the future)
    def batch_cal_mlp(self,embedding_p, concat_vec_mlp_2, name):
        if self.layer_num == 0:
            h_2 = tf.tile(tf.expand_dims(self.h, axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*1 -> 1*2K*1 -> n*2K*1
            return tf.squeeze(tf.matmul(concat_vec_mlp_2, h_2), axis=2, name=name)  # BatchMatMul: n*M*2K n*2K*1 -> n*M*1 -> n*M
        elif self.layer_num == 1:
            W_FC_2 = tf.tile(tf.expand_dims(self.W_FC, axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*K -> 1*2K*K -> n*2K*K
            b_FC_2 = self.b_FC
            h_2 = tf.tile(tf.expand_dims(self.h, axis=0), [tf.shape(embedding_p)[0], 1, 1])  # K*1 -> 1*K*1 -> n*K*1
            fc_2 = tf.nn.relu(tf.matmul(concat_vec_mlp_2, W_FC_2) + b_FC_2)  # n*M*2K n*2K*K
            return tf.squeeze(tf.matmul(fc_2, h_2), axis=2, name=name)
        else:
            fc_2 = []
            for i in range(self.layer_num):
                if i == 0:
                    W_FC_2 = tf.tile(tf.expand_dims(self.W_FC[i], axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*K -> 1*2K*K -> n*2K*K
                    b_FC_2 = self.b_FC[i]
                    fc_2.append(tf.nn.relu(tf.matmul(concat_vec_mlp_2, W_FC_2) + b_FC_2))  # n*M*2K n*2K*K
                else:
                    W_FC_2 = tf.tile(tf.expand_dims(self.W_FC[i], axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*K -> 1*2K*K -> n*2K*K
                    b_FC_2 = self.b_FC[i]
                    fc_2.append(tf.nn.relu(tf.matmul(fc_2[i - 1], W_FC_2) + b_FC_2))
            h_2 = tf.tile(tf.expand_dims(self.h, axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*1 -> 1*2K*1 -> n*2K*1
            return tf.squeeze(tf.matmul(fc_2[i], h_2), axis=2, name=name)

    def _create_loss(self):
        embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
        embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)

        # all logits
        concat_vec_mlp_2 = tf.concat([tf.tile(tf.expand_dims(embedding_p, axis=1), [1, self.num_items, 1]),
                                      tf.tile(tf.expand_dims(self.embedding_Q, axis=0), [tf.shape(embedding_p)[0], 1, 1])], 2) # n*M*2K
        self.all_logits = self.batch_cal_mlp(embedding_p, concat_vec_mlp_2, 'all_logits')

        # sampled logits
        embedding_q_batch = tf.nn.embedding_lookup(self.embedding_Q, self.candidates_neg)  # n*C*K
        concat_vec_mlp_3 = tf.concat([tf.tile(tf.expand_dims(embedding_p, axis=1), [1, tf.shape(self.candidates_neg)[1], 1]),
                                      embedding_q_batch], 2)  # n*C*2K
        self.sampled_logits = self.batch_cal_mlp(embedding_p, concat_vec_mlp_3, 'sampled_logits')

        # list logits
        embedding_q_list = tf.nn.embedding_lookup(self.embedding_Q, self.candidates_reclist)  # n*C*K
        concat_vec_mlp_4 = tf.concat(
            [tf.tile(tf.expand_dims(embedding_p, axis=1), [1, tf.shape(self.candidates_reclist)[1], 1]),
             embedding_q_list], 2)  # n*C*2K
        self.list_logits = self.batch_cal_mlp(embedding_p, concat_vec_mlp_4, 'list_logits')

        # Loss
        self.regularizer = tf.contrib.layers.l2_regularizer(self.lambda_bilinear)
        self.loss_reg_em = self.regularizer(embedding_p) + self.regularizer(embedding_q)
        if self.layer_num == 1:
            self.loss_reg_w = self.regularizer(self.W_FC)
        else:
            self.loss_reg_w = 0
            for i in range(0, self.layer_num):
                self.loss_reg_w += self.regularizer(self.W_FC[i])
        self.loss_reg = self.loss_reg_w + self.loss_reg_em

    def reg_loss_em(self, input, type):
        if type == "user":
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, input), 1)
            return self.regularizer(embedding_p)
        elif type == "item":
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, input), 1)
            return self.regularizer(embedding_q)

        # if self.layer_num == 0:
        #     h_2 = tf.tile(tf.expand_dims(self.h, axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*1 -> 1*2K*1 -> n*2K*1
        #     self.all_logits = \
        #         tf.squeeze(tf.matmul(concat_vec_mlp_2, h_2), axis=2, name='all_logits')  # BatchMatMul: n*M*2K n*2K*1 -> n*M*1 -> n*M
        # elif self.layer_num == 1:
        #     W_FC_2 = tf.tile(tf.expand_dims(self.W_FC, axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*K -> 1*2K*K -> n*2K*K
        #     b_FC_2 = self.b_FC
        #     h_2 = tf.tile(tf.expand_dims(self.h, axis=0), [tf.shape(embedding_p)[0], 1, 1])  # K*1 -> 1*K*1 -> n*K*1
        #     fc_2 = tf.nn.relu(tf.matmul(concat_vec_mlp_2, W_FC_2) + b_FC_2)  # n*M*2K n*2K*K
        #     self.all_logits = \
        #         tf.squeeze(tf.matmul(fc_2, h_2), axis=2, name='all_logits')
        # else:
        #     fc_2 = []
        #     for i in range(self.layer_num):
        #         if i == 0:
        #             W_FC_2 = tf.tile(tf.expand_dims(self.W_FC[i], axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*K -> 1*2K*K -> n*2K*K
        #             b_FC_2 = self.b_FC[i]
        #             fc_2.append(tf.nn.relu(tf.matmul(concat_vec_mlp_2, W_FC_2) + b_FC_2))  # n*M*2K n*2K*K
        #         else:
        #             W_FC_2 = tf.tile(tf.expand_dims(self.W_FC[i], axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*K -> 1*2K*K -> n*2K*K
        #             b_FC_2 = self.b_FC[i]
        #             fc_2.append(tf.nn.relu(tf.matmul(fc_2[i - 1], W_FC_2) + b_FC_2))
        #     h_2 = tf.tile(tf.expand_dims(self.h, axis=0), [tf.shape(embedding_p)[0], 1, 1])  # 2K*1 -> 1*2K*1 -> n*2K*1
        #     self.all_logits = \
        #         tf.squeeze(tf.matmul(fc_2[i], h_2), axis=2, name='all_logits')

    def save_model(self, sess, filename):
        if self.layer_num == 0:
            param = sess.run([self.embedding_P, self.embedding_Q, self.h])
        elif self.layer_num == 1:
            param = sess.run([self.embedding_P, self.embedding_Q, self.h])
            Wb = sess.run([self.W_FC, self.b_FC])
            param.append(Wb)
        else:
            param = sess.run([self.embedding_P, self.embedding_Q, self.h])
            Wb = sess.run([self.W_FC, self.b_FC])
            for i in range(len(Wb[0])):
                param.append([ Wb[0][i], Wb[1][i] ])
        cPickle.dump(param, open(filename, 'w'))

class GMF:
    def __init__(self, num_users, num_items, embed_size, regs, use_pretrain=False, param=None):
        self.num_items = num_items
        self.num_users = num_users
        self.embedding_size = embed_size
        self.param = param
        self.lambda_bilinear = regs[0]
        self.gamma_bilinear = regs[1]
        self.use_pretrain = use_pretrain

    def _create_placeholders(self, placeholders):
        with tf.name_scope("input_data"):
            self.user_input = placeholders[0]
            self.item_input = placeholders[1]
            self.candidates_neg = placeholders[2]
            self.candidates_reclist = placeholders[3]

    def _create_variables(self):
        with tf.name_scope("embedding"):
            if self.use_pretrain:
                self.embedding_P = tf.Variable(self.param[0], name='embedding_P')
                self.embedding_Q = tf.Variable(self.param[1], name='embedding_Q')
                self.h = tf.Variable(self.param[2], name='h')
            else:
                self.embedding_P = tf.Variable(
                    tf.truncated_normal(shape=[self.num_users, self.embedding_size], mean=0.0, stddev=0.01),
                                        name='embedding_P', dtype=tf.float32)  # (users, embedding_size)
                self.embedding_Q = tf.Variable(
                    tf.truncated_normal(shape=[self.num_items, self.embedding_size], mean=0.0, stddev=0.01),
                                        name='embedding_Q', dtype=tf.float32)  # (items, embedding_size)
                # Lecun's Uniform Init
                self.h = tf.Variable(tf.random_uniform([self.embedding_size, 1], minval=-tf.sqrt(3 / self.embedding_size),
                                                   maxval=tf.sqrt(3 / self.embedding_size)), name='h')
            self.model_params = [self.embedding_P, self.embedding_Q, self.h]

    def _create_inference(self, item_input):
        with tf.name_scope("inference"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)  # (b, embedding_size)
            # no sigmoid for BPR loss
            return tf.matmul(embedding_p * embedding_q, self.h, name='output')
            # return embedding_p, embedding_q, tf.matmul(embedding_p * embedding_q, self.h, name='output')
            # return tf.sigmoid(
            #     tf.matmul(self.embedding_p * self.embedding_q, self.h), name = 'output')  # (b, embedding_size) * (embedding_size, 1)

    def _create_feature(self, item_input):
        with tf.name_scope("feature"):
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, item_input), 1)
            return embedding_p * embedding_q

    def _create_loss(self):
        embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
        embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, self.item_input), 1)
        self.all_logits = tf.matmul(tf.reshape(self.h,[1,-1])*embedding_p, self.embedding_Q,
                                    transpose_a=False, transpose_b=True, name='all_logits')

        self.regularizer = tf.contrib.layers.l2_regularizer(self.lambda_bilinear)
        if self.lambda_bilinear != 0:
            self.loss_reg = self.regularizer(embedding_p) + self.regularizer(embedding_q)
        else:
            self.loss_reg = 0

        embedding_q_batch = tf.nn.embedding_lookup(self.embedding_Q, self.candidates_neg) # n*C*K
        embedding_p_batch = tf.expand_dims(tf.reshape(self.h,[1,-1])
                                           *tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
                                           , axis=2) # n*K*1
        self.sampled_logits = tf.squeeze(tf.matmul(embedding_q_batch, embedding_p_batch), axis=2) # n*C

        embedding_q_list = tf.nn.embedding_lookup(self.embedding_Q, self.candidates_reclist)  # n*C*K
        embedding_p_list = tf.expand_dims(tf.reshape(self.h, [1, -1])
                                           * tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, self.user_input), 1)
                                           , axis=2)  # n*K*1
        self.list_logits = tf.squeeze(tf.matmul(embedding_q_list, embedding_p_list), axis=2)  # n*C

        self.loss_reg_w = 0

    def reg_loss_em(self, input, type):
        if type == "user":
            embedding_p = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_P, input), 1)
            return self.regularizer(embedding_p)
        elif type == "item":
            embedding_q = tf.reduce_sum(tf.nn.embedding_lookup(self.embedding_Q, input), 1)
            return self.regularizer(embedding_q)

    def save_model(self, sess, filename):
        param = sess.run([self.embedding_P, self.embedding_Q, self.h])
        cPickle.dump(param, open(filename, 'w'))

class ItemPop:
    def __init__(self, num_users, num_items, items_score, args, reclist_len=1):
        self.num_items = num_items
        self.num_users = num_users
        self.items_score = items_score # an np.array, score bigger -> more popular
        self.Lreclist = reclist_len

    def _create_placeholders(self):
        with tf.name_scope("input_data"):
            self.user_input = tf.placeholder(tf.int32, shape=[None, 1], name="user_input")
            self.item_input = tf.placeholder(tf.int32, shape=[None, 1], name="item_input")
            self.candidates_reclist = tf.placeholder(tf.int32, shape=[None, self.Lreclist], name="candidates_reclist")

    def _create_variables(self):
        with tf.name_scope("embedding"):
            self.items_score_tf = tf.convert_to_tensor(self.items_score)

    def _create_inference(self, item_input, name):
        with tf.name_scope("inference"):
            return tf.reduce_sum(tf.nn.embedding_lookup(self.items_score_tf, item_input), 1)

    def _create_loss(self):
        with tf.name_scope("loss"):
            self.output = self._create_inference(self.item_input, 'output')
            self.all_rating = tf.tile(tf.reshape(self.items_score_tf,[1,-1]),[tf.shape(self.user_input)[0],1])
            self.list_rating = tf.squeeze(tf.gather(self.items_score_tf, self.candidates_reclist),axis=2)
            _, list_indices = tf.nn.top_k(self.list_rating, k=self.Lreclist)
            _, self.list_order = tf.nn.top_k(-list_indices, k=self.Lreclist)

    def build_graph(self):
        self._create_placeholders()
        self._create_variables()
        self._create_loss()