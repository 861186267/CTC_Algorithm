#! python 
# @Time    : 17-9-26
# @Author  : kay
# @File    : eva_template_standard.py
# @E-mail  : 861186267@qq.com
# @Function:

from glob import iglob
import sys

sys.path.append('../')
sys.dont_write_bytecode = True

import datetime
from utils.utils import *
from model.ori_bilstm import ExtractorLSTM
from tensorflow.python.platform import flags
from decoder.ctc_beam_search import *
from decoder.edit_distance import *
from model.rew_bilstm import *


np.set_printoptions(threshold=10000000000, linewidth=10000000000000)

flags.DEFINE_string('utter_dataset', 'utterances', 'set the template dataset')

flags.DEFINE_string('level', 'phn', 'set the task level, phn, cha, or seq2seq, seq2seq will be supported soon')
flags.DEFINE_string('model', 'ExtractorLSTM', 'set the model to use, DBiRNN, BiRNN, ResNet..')

flags.DEFINE_integer('batch_size', 1, 'set the batch size')
flags.DEFINE_integer('num_hidden', 128, 'set the hidden size of rnn cell')
flags.DEFINE_integer('num_feature', 40, 'set the size of input feature')
flags.DEFINE_integer('num_classes', 70, 'set the number of output classes')

flags.DEFINE_string('datadir', '/home/kay/Desktop/ctc/resource/code/python/data', 'set the pos root directory')
flags.DEFINE_string('logdir', '/home/kay/Desktop/ctc/resource/code/python/data', 'set the log directory')

FLAGS = flags.FLAGS

utter_dataset = FLAGS.utter_dataset
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

logfile = os.path.join(loggingdir, str(datetime.datetime.strftime(datetime.datetime.now(),
                                                                  '%Y-%m-%d %H:%M:%S') + '.txt').replace(' ',
                                                                                                         '').replace(
    '/', ''))


def get_templates():
    """
    function: get templates saved by users
    :return: templates
    """
    template_dir = os.path.join(datadir, level, 'templates_standard')
    templates = []
    for index, temp_path in enumerate(iglob(os.path.join(template_dir, '**/**.npy'), recursive=True)):
        temp = np.load(temp_path)
        templates.append(temp)

    return templates


def get_utterances(datadir, level, utter_dataset):
    """
    function: get utterances going to test
    :param datadir: the directory of utterances
    :param level: the default is phn
    :param utter_dataset: the subdirectory of utterances 
    :return: the path list of utterances feature and labels
    """
    feature_dirs = []
    label_dirs = []
    for idx, wav_path in enumerate(
            iglob(os.path.join(datadir, level, utter_dataset, 'feature', '**/**.npy'), recursive=True)):
        feature_dirs.append(wav_path)
        lab_path = wav_path.replace('feature', 'label')
        label_dirs.append(lab_path)

    return os.path.join(datadir, level, utter_dataset, 'feature', '**/**.npy'), os.path.join(datadir, level,
                                                                                             utter_dataset, 'feature',
                                                                                             '**/**.npy')


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

        feature_dir = os.path.join(datadir, level, utter_dataset, 'feature')
        label_dir = os.path.join(datadir, level, utter_dataset, 'label')

        print('feature_dir:', feature_dir)
        print('label_dir:', label_dir)

        batchedData, maxTimeSteps, totalN = load_batched_data(feature_dir, label_dir, batch_size, level)
        print('len(batchedData):', len(batchedData))

        feva_result = os.path.join(resultdir, 'eva_utter.txt')
        if os.path.exists(feva_result):
            os.remove(feva_result)

        passcount = 0
        standard = 0
        phnslist = get_phns_list(num_classes)

        with tf.Session(graph=model.graph) as sess:
            for batch in batchedData:
                batchInputs, batchTargetSparse, batchSeqLengths = batch

                # params_path = os.path.join(os.getcwd(), '../', 'parameter', 'ctc_parameters.txt')
                # print('params_path:', params_path)
                # fparams = open(params_path, 'r')
                # params = []
                # for param in fparams.readlines():
                #     print('param:', np.shape(param))
                #     params.append(param)

                ckpt = tf.train.get_checkpoint_state(savedir)
                model.saver.restore(sess, ckpt.model_checkpoint_path)

                params = sess.run(model.var_trainable_op)
                logits2d = QbyENetwork(batchInputs, params, num_hidden, num_classes)

                beam_result_log = ctc_beam_search_decoder_log(
                    probs_seq=logits2d,
                    beam_size=1,
                    vocabulary=phnslist,
                    blank_id=len(phnslist),
                    cutoff_prob=1.0)

                print(beam_result_log)

                pres_ = [int(item) for item in beam_result_log[0][1].split('_')[1:]]
                pres_list = list(pres_)

                print('pres_:', pres_)
                print('pres_list:', pres_list)

                spos = 0  # start pos remove silence
                epos = len(pres_list)  # end pos remove silence
                if epos == 0:
                    print('pres_list is null')
                    continue

                if int(pres_list[0]) == 0:
                    spos = 1
                if int(pres_list[epos - 1]) == 0:
                    epos -= 1

                pre_ori = pres_[spos:epos]
                pre_list_ori = pres_list[spos:epos]
                pre_len = len(pre_list_ori)

                whone = 0
                tolerate = 0.35  # the toleration degree of length of prediction
                err = 0
                success = False
                templates = get_templates()
                for tag, temp in enumerate(templates):
                    print('passcount:', passcount)
                    # for temp in templates:
                    whone = tag

                    temp_ = temp
                    temp_list = temp

                    spos_ = 0  # start pos remove silence
                    epos_ = len(temp_list)  # end pos remove silence

                    print('spos_:', spos_)
                    print('epos_:', epos_)
                    print('temp_list:', temp_list)

                    if int(temp_list[0]) == 0:
                        spos_ = 1

                    if int(temp_list[epos_ - 1]) == 0:
                        epos_ -= 1

                    tmp_ori = temp_[spos_:epos_]
                    tmp_list_ori = temp_list[spos_:epos_]
                    tmp_len = len(tmp_list_ori)

                    # standard is the threshold
                    tole_len = np.ceil(tmp_len / 2) - 1
                    standard = tole_len / tmp_len  # + 0.001

                    # the prediction is far shorter than template over the toleration
                    if pre_len <= tmp_len * tolerate:
                        continue

                    # the predication is a bit shorter than template under the toleration
                    elif pre_len <= tmp_len:
                        err = edit_distance(pre_ori, tmp_ori)
                        if err < standard:
                            success = True
                            passcount += 1
                            break
                        else:
                            continue

                    # the prediction is longer than template
                    else:
                        # whether target is in prediction or not
                        if (''.join(map(repr, tmp_list_ori)) in ''.join(map(repr, pre_list_ori))):
                            passcount += 1
                            break

                        else:
                            # find the same element in prediction and template, then do comparison
                            for i, atom in enumerate(tmp_list_ori):
                                try:
                                    idx = pre_list_ori.index(atom)
                                    start = idx - i
                                    if start < 0:
                                        start = 0

                                    end = idx + tmp_len - i
                                    if end > pre_len:
                                        end = pre_len

                                    err = edit_distance(pres_[start + 1:end + 1], temp_[1:tmp_len + 1])

                                    if err < standard:
                                        success = True
                                        passcount += 1
                                        break
                                    else:
                                        continue

                                except ValueError:
                                    continue

                    if success:
                        break

                with open(feva_result, 'a') as result:
                    result.write(output_to_sequence(templates[whone], type=level) + '\n')
                    result.write(output_to_sequence(pres_list, type=level) + '\n')
                    result.write('standard:' + str(standard) + ', pErr:' + str(err) + ' ' + str(success) + '\n')
                    result.write('\n')
                    result.close()

        sess.close()
        print('passcount:', passcount)
        print('totalN:', totalN)
        print('percent:', passcount / totalN)


if __name__ == '__main__':
    runner = Runner()
    runner.run()
