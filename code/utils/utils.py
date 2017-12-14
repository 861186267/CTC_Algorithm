#! python 
# @Time    : 17-9-8
# @Author  : kay
# @File    : utils.py
# @E-mail  : 861186267@qq.com
# @Function:

import sys

sys.path.append('../')
sys.dont_write_bytecode = True

import os
import numpy as np
from process.map_data import *


def load_batched_data(mfccPath, labelPath, batchSize, level):
    """returns 3-element tuple: batched data (list), maxTimeLength (int), and
       total number of samples (int)"""
    return data_lists_to_batches([np.load(os.path.join(mfccPath, fn)) for fn in os.listdir(mfccPath)],
                                 [np.load(os.path.join(labelPath, fn)) for fn in os.listdir(labelPath)],
                                 batchSize, level) + \
           (len(os.listdir(mfccPath)),)


def count_params(model, mode='trainable'):
    """ count all parameters of a tensorflow graph
    """
    if mode == 'all':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_op])
    elif mode == 'trainable':
        num = np.sum([np.product([xi.value for xi in x.get_shape()]) for x in model.var_trainable_op])
    else:
        raise TypeError('mode should be all or trainable.')
    print('number of ' + mode + ' parameters: ' + str(num))

    return num


def output_to_sequence(lmt, type='phn'):
    """ convert the output into sequences of characters or phonemes
    """
    phn = get_shrink_phn()

    # here, we only print the first sequence of batch
    indexes = lmt  # here, we only print the first sequence of batch
    if type == 'phn':
        seq = []
        for ind in indexes:
            if ind == len(phn):
                pass
            else:
                seq.append(phn[ind])
        seq = ' '.join(seq)
        return seq
    else:
        raise TypeError('mode should be phoneme')


def list_to_sparse_tensor(targetList, level):
    """ turn 2-D List to SparseTensor
    """
    indices = []  # index
    vals = []  # value
    phn = get_phn()
    mapping_phn = get_mapping_phn()
    shrink_phn = get_shrink_phn()

    if level == 'phn':
        """
        for phn level, we should collapse 138 labels into 69 labels before scoring

        Reference:
          Heterogeneous Acoustic Measurements and Multiple Classifiers for Speech Recognition(1986), 
            Andrew K. Halberstadt, https://groups.csail.mit.edu/sls/publications/1998/phdthesis-drew.pdf
        """
        # print('targetList:', targetList)
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                if val < len(phn) and (phn[val] in mapping_phn.keys()):
                    val = shrink_phn.index(mapping_phn[phn[val]])
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(0)[1] + 1]  # shape

        return (np.array(indices), np.array(vals), np.array(shape))

    else:
        # support seq2seq in future here
        raise ValueError('Invalid level: %s' % str(level))


def data_lists_to_batches(inputList, targetList, batchSize, level):
    """ padding the input list to a same dimension, integrate all data into batchInputs
    """
    assert len(inputList) == len(targetList)
    # dimensions of inputList:batch*39*time_length

    nFeatures = inputList[0].shape[0]
    maxLength = 0
    for inp in inputList:
        # find the max time_length
        maxLength = max(maxLength, inp.shape[1])

    # randIxs is the shuffled index from range(0,len(inputList))
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
        # batchSeqLengths store the time-length of each sample in a mini-batch
        batchSeqLengths = np.zeros(batchSize)

        # randIxs is the shuffled index of input list
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]

        batchInputs = np.zeros((maxLength, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
            # padSecs is the length of padding
            padSecs = maxLength - inputList[origI].shape[1]
            # numpy.pad pad the inputList[origI] with zeos at the tail
            batchInputs[:, batchI, :] = np.pad(inputList[origI].T, ((0, padSecs), (0, 0)), 'constant',
                                               constant_values=0)
            # target label
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, list_to_sparse_tensor(batchTargetList, level), batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxLength)


def get_phns_list(num_classes):
    """
    :return: the index of every phonemes
    """
    phns_list = []
    for item in range(num_classes - 1):
        phns_list.append(str(item))

    return phns_list


def check_path_exists(path):
    """ check a path exists or not
    """
    if isinstance(path, list):
        for p in path:
            if not os.path.exists(p):
                os.makedirs(p)
    else:
        if not os.path.exists(path):
            os.makrdirs(path)


class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
