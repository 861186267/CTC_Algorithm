#! python 
# @Time    : 17-12-6
# @Author  : kay
# @File    : generate_params.py
# @E-mail  : 861186267@qq.com
# @Function:

import sys

sys.path.append('../')
sys.dont_write_bytecode = True

from utils.utils import *
from model.ori_bilstm import ExtractorLSTM
from tensorflow.python.platform import flags
import numpy as np
import tensorflow as tf

flags.DEFINE_string('level', 'phn', 'set the task level, phn, cha, or seq2seq, seq2seq will be supported soon')
flags.DEFINE_string('model', 'ExtractorLSTM', 'set the model to use, DBiRNN, BiRNN, ResNet..')

flags.DEFINE_integer('batch_size', 1, 'set the batch size')
flags.DEFINE_integer('num_hidden', 128, 'set the hidden size of rnn cell')
flags.DEFINE_integer('num_feature', 40, 'set the size of input feature')
flags.DEFINE_integer('num_classes', 70, 'set the number of output classes')

flags.DEFINE_string('datadir', '/home/kay/Desktop/ctc/resource/code/python/data', 'set the pos root directory')
flags.DEFINE_string('logdir', '/home/kay/Desktop/ctc/resource/code/python/data', 'set the log directory')

FLAGS = flags.FLAGS
level = FLAGS.level
model_fn = ExtractorLSTM

batch_size = FLAGS.batch_size
num_hidden = FLAGS.num_hidden
num_feature = FLAGS.num_feature
num_classes = FLAGS.num_classes
datadir = FLAGS.datadir

logdir = FLAGS.logdir
savedir = os.path.join(logdir, level, 'save')

param_path = 'ctc_parameters.npy'
if os.path.exists(param_path):
    os.remove(param_path)

# fparas = open(param_path, 'wt')
np.set_printoptions(threshold=10000000000, linewidth=10000000000000)


class Runner(object):
    def _default_configs(self):
        return {'model_fn': model_fn,
                'batch_size': batch_size,
                'num_hidden': num_hidden,
                'num_feature': num_feature,
                'num_classes': num_classes}

    def run(self):
        # load pos
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        model = model_fn(args, 1)

        with tf.Session(graph=model.graph) as sess:
            # restore from stored model
            ckpt = tf.train.get_checkpoint_state(savedir)
            if ckpt and ckpt.model_checkpoint_path:
                print('ckpt.model_checkpoint_path:', ckpt.model_checkpoint_path)
                print('Model restored from:' + savedir)
                model.saver.restore(sess, ckpt.model_checkpoint_path)

                params = sess.run(model.var_trainable_op)
                np.save(param_path, params)
                # fparas.write(str(params))
            print('params.shape:', np.shape(params))
            print('tf.trainable_variables:', tf.trainable_variables())

        sess.close()
        # fparas.close()


if __name__ == '__main__':
    runner = Runner()
    runner.run()
