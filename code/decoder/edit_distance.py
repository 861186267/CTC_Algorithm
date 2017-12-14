#! python 
# @Time    : 17-12-7
# @Author  : kay
# @File    : edit_distance.py
# @E-mail  : 861186267@qq.com
# @Function:

from utils.utils import get_shrink_phn, get_rhythm_phn
import numpy as np


def edit_distance(predict, temp, norm=True):
    """
    Calculate optimal warping path cost.
    :param predict_ori: test sequence (prediction)
    :param temp_ori: reference sequence (template)
    :param norm: normalization
    :return: cost, global cost matrix
    """
    rows = len(predict)
    cols = len(temp)

    cost_matrix = np.zeros((rows + 1, cols + 1))

    cost_matrix[:, 0] = np.inf
    cost_matrix[0, :] = np.inf
    cost_matrix[0, 0] = 0

    for i in range(rows):
        for j in range(cols):
            cost = edit_dist(predict, temp, i, j)
            cost_matrix[i + 1, j + 1] = cost + min(cost_matrix[i, j + 1],
                                                   cost_matrix[i + 1, j],
                                                   cost_matrix[i, j])

    if norm:
        return cost_matrix[rows, cols] / len(temp)
    else:
        return cost_matrix[rows, cols]


def edit_dist(predict, temp, i, j):
    """
    :param temp: the sequence of template
    :param predict: the sequence of prediction
    :param i: the indices of template
    :param j: the indices of prediction
    :return: 
    """
    shrink_phns = get_shrink_phn()
    rhyme_phns = get_rhythm_phn()

    if temp[j] == predict[i]:
        dist = 0

    else:
        if shrink_phns[temp[j]] in rhyme_phns:
            dist = 1
        else:
            dist = 1

    return dist
