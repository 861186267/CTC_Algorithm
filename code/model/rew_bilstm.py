#! python
# @Time    : 17-12-7
# @Author  : kay
# @File    : rew_bilstm.py
# @E-mail  : 861186267@qq.com
# @Function: the network calculation process after get parameters(model)

import numpy as np
from tensorflow.python.ops import nn_ops, array_ops, math_ops
import tensorflow as tf


def basicLSTM(inputs, state, weights, biases):
    '''
    :param inputs: the inputs data e.g.(1, 40) in 1st hidden; (1, 128) in 2st hidden
    :param state: the tuple of (c, h)
    :param weights: the corresponding weights from model
    :param biases: the corresponding biases from model 
    :return: 
    '''
    c, h = state
    inputs_h = array_ops.concat((inputs, h), axis=1)

    w_b_Value = nn_ops.bias_add(math_ops.matmul(inputs_h, weights), biases)
    i, j, f, o = array_ops.split(w_b_Value, num_or_size_splits=4, axis=1)

    c_new = math_ops.sigmoid(i) * math_ops.tanh(j) + math_ops.sigmoid(f + 1) * c
    h_new = math_ops.sigmoid(o) * math_ops.tanh(c_new)

    return h_new, (c_new, h_new)


def dynamic_rnn(frame, state, params, param_id):
    '''
    :param frame: per frame data e.g.(1, 40) in 1st hidden; (1, 128) in 2st hidden
    :param state: the tuple of (c, h)
    :param params: the list parameters
    :param param_id: the index of which part of parameters to start to get
    :return: the tensor of h and state
    '''

    weights = tf.convert_to_tensor(params[param_id], dtype=tf.float32)
    biases = tf.convert_to_tensor(params[param_id + 1], dtype=tf.float32)
    h, state = basicLSTM(frame, state, weights, biases)

    return h, state


def biLSTM(inputs, params, num_hidden, param_id):
    '''
    :param inputs: the inputs data e.g.(seqLen, 1, 40), seqLen is time step which represents the number of frames
    :param params: the list parameters
    :param num_hidden: the number of hidden cells per layer, e.g.128
    :param param_id: the index of which part of parameters to start to get
    :return: the sum the forward and backward fruits as output
    '''
    c = array_ops.zeros((1, num_hidden))  # initial cell state zeros tensor shape (1, 128) for the first time
    h = array_ops.zeros((1, num_hidden))  # initial output state zeros tensor shape (1, 128) for the first time
    state = c, h

    forward_inputs = inputs
    forward_state = state
    forward_tmp = []
    for frame in forward_inputs.eval():
        forward_h, forward_state = dynamic_rnn(frame, forward_state, params, param_id)
        forward_tmp.append(forward_h)
    forward_h = tf.stack(forward_tmp)

    # reverse data when do backward LSTM
    backward_inputs = array_ops.reverse_sequence(input=inputs, seq_lengths=(inputs.get_shape().as_list()[0],),
                                                 seq_axis=0, batch_dim=1)
    backward_state = state
    backward_tmp = []
    for frame in backward_inputs.eval():
        backward_h, backward_state = dynamic_rnn(frame, backward_state, params, param_id + 2)
        backward_tmp.append(backward_h)
    backward_tmp.reverse()
    backward_h = tf.stack(backward_tmp)

    # sum the forward and backward fruits as output
    hidden = tf.reduce_sum((forward_h, backward_h), axis=0)

    return hidden


def QbyENetwork(inputs, params, num_hidden, num_classes):
    '''
    :param inputs: the inputs data e.g.(seqLen, 1, 40), seqLen is time step which represents the number of frames
    :param params: the list parameters
    :param num_hidden: the number of hidden cells per layer, e.g.128
    :return: the probability classification set, e.g. (seqLen, 70)
    '''
    inputs_list = tf.convert_to_tensor(inputs)
    num_hidden_layer = 2
    param_id = 2
    for _ in range(num_hidden_layer):
        print('param_id:', param_id)
        inputs_list = biLSTM(inputs_list, params, num_hidden, param_id)
        param_id += 4

    outputXrs = tf.reshape(inputs_list, [-1, num_hidden])  # (seqlen, 128)
    shape = inputs_list.get_shape().as_list()
    output_list = tf.split(outputXrs, shape[0], 0)  # (seqlen,)

    fbHrs = [tf.reshape(t, [1, num_hidden]) for t in output_list]  # (1, 0)

    W_out = params[0]  # (128, 70)
    b_out = params[1]  # (70, 1)

    logits = [tf.matmul(t, W_out) + b_out for t in fbHrs]

    logits3d = tf.stack(logits).eval()  # (seqLen, 1, 70)

    logits2d = np.reshape(logits3d, [-1, num_classes])

    print('logits2d:', np.shape(logits2d))

    return logits2d
