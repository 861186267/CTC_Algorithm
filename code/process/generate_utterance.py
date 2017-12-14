#! python 
# @Time    : 17-9-7
# @Author  : kay
# @File    : generate_utterance.py
# @E-mail  : 861186267@qq.com
# @Function:

from __future__ import print_function
from __future__ import unicode_literals

import sys

sys.path.append('../')

from process.langconv import *
import argparse
import os
import scipy.io.wavfile as wav
from extractor.features import *
from process.map_data import *
from glob import iglob
np.set_printoptions(threshold=1000000)

folds = get_folds()
np.random.shuffle(folds)
phn = get_phn()
print('len(phn):', len(phn))
print('phn:', phn)

dictionary_path = './data/align_lexicon.txt'
dict_words = get_dictionary(dictionary_path)
print('len(dict_words):', len(dict_words))


def wav2feature(root_dir, save_dir, dataset, win_len, win_step, feature_len):
    for idx, wav_path in enumerate(iglob(os.path.join(root_dir, '**/**.wav'), recursive=True)):
        prefix = wav_path[:-4].split('/')[-1]
        lab_path = wav_path[:-4] + '.lab'
        content = open(lab_path, 'r').readlines()[0].strip('\n')
        label = []

        rate, signal = wav.read(wav_path)

        # add gaussian noise
        signal = np.array(signal, dtype=np.float64)
        noise = np.random.normal(size=np.shape(signal))
        signal += noise
        feat = get_filterbank_energies(signal=signal, rate=rate, frame_length=win_len, frame_step=win_step,
                                       window_function=lambda x: np.ones((x,)),
                                       filters_count=feature_len, fft_size=512, low_freq=100,
                                       high_freq=6800, log=True)

        feat = scale(feat)
        feat = np.transpose(feat)
        print('filterbank_energies transpose:', feat)

        label.append(phn.index('sil'))
        for key in content:
            if key in dict_words:
                if key != '!SIL':
                    value = dict_words[key]
                    print(value)
                    elements = value.strip(' ').split(' ')
                    print(elements)
                    for atom in elements:
                        label.append(phn.index(atom.split('_')[0]))
                else:
                    for it in key:
                        try:
                            value = dict_words[it]
                        except KeyError:
                            if is_chinese(it):
                                # convert traditional chinese to sample chinese
                                it = Converter('zh-hans').convert(it)
                                print('it:', it)
                                value = dict_words[it]
                            else:
                                value = dict_words[it.lower()]

                        elements = value.strip(' ').split(' ')
                        for atom in elements:
                            label.append(phn.index(atom.split('_')[0]))

        label.append(phn.index('sil'))

        label_dir = os.path.join(save_dir, 'phn', dataset, 'label')
        feat_dir = os.path.join(save_dir, 'phn', dataset, 'feature')

        if not os.path.isdir(label_dir):
            os.makedirs(label_dir)
        if not os.path.isdir(feat_dir):
            os.makedirs(feat_dir)

        print('np.shape(feat):', np.shape(feat))
        print('label:', label)
        feat_fn = os.path.join(feat_dir, prefix + '.npy')
        np.save(feat_fn, feat)

        lab_fn = os.path.join(label_dir, prefix + '.npy')
        np.save(lab_fn, label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='dataset_preprocess',
                                     description='Script to preprocess dataset')
    parser.add_argument("-path", help="Directory of dataset", type=str,
                        default='/home/kay/Desktop/ctc/resource/code/python/data/utterances')

    parser.add_argument("-save", help="Directory where preprocessed arrays are to be saved",
                        type=str, default='/home/kay/Desktop/ctc/resource/code/python/data')

    parser.add_argument("-dataset", help="the corresponding dataset", type=str, default='utterances')

    parser.add_argument("-featlen", help='Features length', type=int, default=40)
    parser.add_argument("-winlen", help="specify the window length of feature", type=float, default=0.025)
    parser.add_argument("-winstep", help="specify the window step length of feature", type=float, default=0.01)

    args = parser.parse_args()
    root_dir = args.path
    save_dir = args.save
    dataset = args.dataset

    feature_len = args.featlen
    win_len = args.winlen
    win_step = args.winstep

    if not os.path.isdir(root_dir):
        raise ValueError("Directory does not exist!")

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    wav2feature(root_dir, save_dir, dataset, win_len=win_len, win_step=win_step, feature_len=feature_len)
