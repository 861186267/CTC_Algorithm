#! python 
# @Time    : 17-9-8
# @Author  : kay
# @File    : ori_bilstm.py
# @E-mail  : 861186267@qq.com
# @Function: two layers + biLSTM
# https://r2rt.com/recurrent-neural-networks-in-tensorflow-iii-variable-length-sequences.html

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from model.rnn_cell_impl import BasicLSTMCell
from model.bidyrnn import bidirectional_dynamic_rnn
# from model.bi_lstm import bidirectional_dynamic_rnn

import numpy as np


class ExtractorLSTM(object):
    def __init__(self, args, maxTimeSteps):
        self.args = args
        self.maxTimeSteps = maxTimeSteps
        self.build_graph(args, maxTimeSteps)

    def build_graph(self, args, maxTimeSteps):

        self.graph = tf.Graph()
        with self.graph.as_default():

            # graph input
            self.inputX = tf.placeholder(tf.float32,
                                         shape=(maxTimeSteps, args.batch_size, args.num_feature))  # [maxL,64,40]
            print('inputX.shape:', np.shape(self.inputX))

            self.targetIxs = tf.placeholder(tf.int64)
            self.targetVals = tf.placeholder(tf.int32)
            self.targetShape = tf.placeholder(tf.int64)
            self.targetY = tf.SparseTensor(self.targetIxs, self.targetVals, self.targetShape)
            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size,))

            # write this config to logging
            self.config = {'model': args.model_fn,
                           'batch_size': args.batch_size,
                           'num_hidden': args.num_hidden,
                           'num_feature': args.num_feature,
                           'num_classes': args.num_classes}

            # weights and biases
            self.W_out = tf.Variable(tf.truncated_normal([args.num_hidden, args.num_classes]), name='W_out')
            self.b_out = tf.Variable(tf.truncated_normal([args.num_classes]), name='b_out')

            X = self.inputX
            hidden = None

            # networkl
            for i in range(2):
                scope = 'in_hidden_' + str(i)

                if i == 0:
                    self.forward_cell_1 = BasicLSTMCell(args.num_hidden)
                    self.backward_cell_1 = BasicLSTMCell(args.num_hidden)

                    self.outputs_1, self.states_1 = bidirectional_dynamic_rnn(self.forward_cell_1, self.backward_cell_1,
                                                                inputs=X,
                                                                dtype=tf.float32,
                                                                sequence_length=self.seqLengths,
                                                                scope=scope)

                    self.output_fw_1, self.output_bw_1 = self.outputs_1
                    print('self.output_fw_1:', self.output_fw_1)
                    print('self.output_bw_1:', self.output_bw_1)
                    self.output_con_1 = tf.concat([self.output_fw_1, self.output_bw_1], 2)
                    print('self.output_con_1.shape:', np.shape(self.output_con_1))  # (1085, 64, 256)
                    shape = self.output_con_1.get_shape().as_list()
                    print('shape.shape:', shape)  # [1085, 64, 256]
                    self.output_con_1 = tf.reshape(self.output_con_1, [shape[0], shape[1], 2, int(shape[2] / 2)])
                    print('self.output_con_1.shape:', np.shape(self.output_con_1))  # (1085, 64, 2, 128)
                    hidden = tf.reduce_sum(self.output_con_1, 2)
                    hidden = tf.contrib.layers.dropout(hidden, keep_prob=1, is_training=False)


                    print(scope + '.shape:', np.shape(hidden))  # (1085, 64, 128)

                else:
                    self.forward_cell_2 = BasicLSTMCell(args.num_hidden)
                    self.backward_cell_2 = BasicLSTMCell(args.num_hidden)

                    self.outputs_2, self.states_2 = bidirectional_dynamic_rnn(self.forward_cell_2, self.backward_cell_2,
                                                                inputs=X,
                                                                dtype=tf.float32,
                                                                sequence_length=self.seqLengths,
                                                                scope=scope)



                    self.output_fw_2, self.output_bw_2 = self.outputs_2
                    print('self.output_fw_2:', self.output_fw_2)
                    print('self.output_bw_2:', self.output_bw_2)
                    self.output_con_2 = tf.concat([self.output_fw_2, self.output_bw_2], 2)
                    print('self.output_con_2.shape:', np.shape(self.output_con_2))  # (1085, 64, 256)
                    shape = self.output_con_2.get_shape().as_list()
                    print('shape.shape:', shape)  # [1085, 64, 256]
                    self.output_con_2 = tf.reshape(self.output_con_2, [shape[0], shape[1], 2, int(shape[2] / 2)])
                    print('self.output_con_2.shape:', np.shape(self.output_con_2))  # (1085, 64, 2, 128)
                    hidden = tf.reduce_sum(self.output_con_2, 2)
                    hidden = tf.contrib.layers.dropout(hidden, keep_prob=1, is_training=False)
                    print(scope + '.shape:', np.shape(hidden))  # (1085, 64, 128)



                X = hidden
                self.test = hidden

            outputXrs = tf.reshape(hidden, [-1, args.num_hidden])
            print('outputXrs.shape:', np.shape(outputXrs))  # (1085X64, 128)
            self.output_list = tf.split(outputXrs, maxTimeSteps, 0)
            print('output_list.shape:', np.shape(self.output_list))  # (1085,)
            self.fbHrs = [tf.reshape(t, [args.batch_size, args.num_hidden]) for t in self.output_list]
            print('fbHrs.shape:', np.shape(self.fbHrs))   # (1085,)

            with tf.variable_scope('out_hidden'):
                logits = [tf.matmul(t, self.W_out) + self.b_out for t in self.fbHrs]

            # optimizer
            self.logits3d = tf.stack(logits)

            print('self.logits3d:', self.logits3d)  # (1085,)

            # initialization
            self.var_trainable_op = tf.trainable_variables()
            self.var_op = tf.global_variables()
            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(self.var_op, max_to_keep=5, keep_checkpoint_every_n_hours=1)

