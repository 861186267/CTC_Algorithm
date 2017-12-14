#! python 
# @Time    : 17-12-4
# @Author  : kay
# @File    : generate_template.py
# @E-mail  : 861186267@qq.com
# @Function:

from __future__ import print_function
from __future__ import unicode_literals

import sys

sys.path.append('../')

import argparse
import scipy.io.wavfile as wav
from extractor.features import *
from glob import iglob
from decoder.ctc_beam_search import *
import sys
import shutil
import tensorflow as tf

sys.path.append('../')
sys.dont_write_bytecode = True

from utils.utils import *
from model.ori_bilstm import ExtractorLSTM

from tensorflow.python.platform import flags

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
resultdir = os.path.join(logdir, level, 'result')
loggingdir = os.path.join(logdir, level, 'logging')
check_path_exists([logdir, savedir, resultdir, loggingdir])

phn = get_phn()
print('len(phn):', len(phn))
print('phn:', phn)


def get_data(root_dir, win_len, win_step, feature_len):
    feats = []
    for i, wave_path in enumerate(iglob(os.path.join(root_dir, '**/**.wav'), recursive=True)):
        rate, signal = wav.read(wave_path)

        # add gaussian noise
        signal = np.array(signal, dtype=np.float64)
        noise = np.random.normal(size=np.shape(signal))
        signal += noise

        feat = get_filterbank_energies(signal=signal, rate=rate, frame_length=win_len, frame_step=win_step,
                                       window_function=lambda x: np.ones((x,)),
                                       filters_count=feature_len, fft_size=512, low_freq=100,
                                       high_freq=6800, log=True)

        feat = scale(feat)
        print('filterbank_energies scale:', np.shape(feat))

        feats.append(feat)

    return feats


class Runner(object):
    def _default_configs(self):

        return {'model_fn': model_fn,
                'batch_size': batch_size,
                'num_hidden': num_hidden,
                'num_feature': num_feature,
                'num_classes': num_classes}

    def run(self, root_dir, dataset, win_len, win_step, feature_len):
        # load pos
        args_dict = self._default_configs()
        args = dotdict(args_dict)
        feats = get_data(root_dir, win_len=win_len, win_step=win_step, feature_len=feature_len)

        idx = 0

        template_txt = os.path.join(resultdir, 'template.txt')

        if os.path.exists(template_txt):
            print('exist template_txt')
            os.remove(template_txt)

        template_dir = os.path.join(logdir, level, dataset)

        if os.path.exists(template_dir):
            shutil.rmtree(template_dir)

        if not os.path.exists(template_dir):
            os.makedirs(template_dir)

        for feat in feats:
            batchedData = feat
            maxTimeSteps = np.shape(feat)[0]
            model = model_fn(args, maxTimeSteps)
            print('len(batchedData):', len(batchedData))

            num_params = count_params(model, mode='trainable')
            all_num_params = count_params(model, mode='all')
            model.config['trainable params'] = num_params
            model.config['all params'] = all_num_params

            print('model.config:', model.config)

            phnslist = get_phns_list(num_classes)

            with tf.Session(graph=model.graph) as sess:
                # restore from stored model
                ckpt = tf.train.get_checkpoint_state(savedir)
                if ckpt and ckpt.model_checkpoint_path:
                    print('ckpt.model_checkpoint_path:', ckpt.model_checkpoint_path)
                    print('Model restored from:' + savedir)
                    model.saver.restore(sess, ckpt.model_checkpoint_path)

                inputX = np.reshape(feat, [maxTimeSteps, -1, num_feature])
                feedDict = {model.inputX: inputX, model.seqLengths: [maxTimeSteps, ]}

                logits3d = sess.run(model.logits3d, feed_dict=feedDict)
                logits2d = np.reshape(logits3d, [-1, num_classes])

                beam_result_log = ctc_beam_search_decoder_log(
                    probs_seq=logits2d,
                    beam_size=1,
                    vocabulary=phnslist,
                    blank_id=len(phnslist),
                    cutoff_prob=1.0)

                print(beam_result_log)

                pre = [int(item) for item in beam_result_log[0][1].split('_')[1:]]
                print('pre:', pre)

                # pre = sess.run(model.predictions, feed_dict=feedDict)
                template_path = os.path.join(template_dir, str(idx) + '.npy')

                np.save(template_path, pre)

                output = output_to_sequence(pre, type=level)

                print('output:', output)

                with open(template_txt, 'a') as result:
                    result.write(str(pre) + '\n')
                    result.write(output + '\n')
                    result.write('\n')

                idx += 1

            sess.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dataset_preprocess',
                                     description='Script to preprocess dataset')
    parser.add_argument("-path", help="Directory of dataset", type=str,
                        default='/home/kay/Desktop/ctc/resource/code/python/data/templates')

    parser.add_argument("-dataset", help="the corresponding dataset", type=str, default='templates')

    parser.add_argument("-featlen", help='Features length', type=int, default=40)
    parser.add_argument("-winlen", help="specify the window length of feature", type=float, default=0.025)
    parser.add_argument("-winstep", help="specify the window step length of feature", type=float, default=0.01)

    args = parser.parse_args()
    dataset = args.dataset

    feature_len = args.featlen
    win_len = args.winlen
    win_step = args.winstep

    root_dir = args.path
    if not os.path.isdir(root_dir):
        print('root_dir:', root_dir)
        raise ValueError("Directory does not exist!")

    runner = Runner()
    runner.run(root_dir, dataset, win_len, win_step, feature_len)
