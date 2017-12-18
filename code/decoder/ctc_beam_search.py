#! python 
# @Time    : 17-12-13
# @Author  : kay
# @File    : ctc_beam_search.py
# @E-mail  : 861186267@qq.com
# @Function:
import numpy as np

FLT64_MIN = np.float64('-inf')


def log_sum_exp(x, y):
    if x == FLT64_MIN:
        return y
    if y == FLT64_MIN:
        return x

    maxvalue = max(x, y)
    z = np.log(np.exp(x - maxvalue) + np.exp(y - maxvalue)) + maxvalue

    return z


def ctc_beam_search_decoder_log(probs_seq, beam_size, vocabulary, blank_id, cutoff_prob):
    '''
    :param probs_seq: the 2D list with length time steps sequence, 
                      each element is a list of normalized probabilities the shape (seqLen, 70)
    :param beam_size: width of beam search
    :param vocabulary: the list of phonemes, this project is 69
    :param blank_id: the id of blank
    :return: Decoding log probability and result string
    '''

    for prob_list in probs_seq:
        if not len(prob_list) == len(vocabulary) + 1:
            raise ValueError("probs dimension mismatched with vocabulary")

    num_time_steps = len(probs_seq)

    # blank_id check
    probs_dim = len(probs_seq[0])
    if not blank_id < probs_dim:
        raise ValueError("blank_id shouldn't be greater than probs dimension")

    ## initialize
    # prefix_set_prev: the set containing selected prefixes
    # log_probs_b_prev: prefixes' probability ending with blank in previous step
    # log_probs_nb_prev: prefixes' probability ending with non-blank in previous step
    prefix_set_prev = {'\t': 0}
    log_probs_b_prev = {'\t': 0}
    log_probs_nb_prev = {'\t': FLT64_MIN}

    # extend prefix in loop
    for time_steps in range(num_time_steps):
        prefix_set_next = {}
        log_probs_b_cur = {}
        log_probs_nb_cur = {}

        probs_idx = list(enumerate(probs_seq[time_steps]))
        cutoff_len = len(probs_idx)

        # if pruning is enabled
        if cutoff_prob < 1.0:
            probs_idx = sorted(probs_idx, key=lambda asd: asd[1], reverse=True)
            cutoff_len = 0
            cum_prob = 0

            for i in range(len(probs_idx)):
                cum_prob += probs_idx[i][1]
                cutoff_len += 1
                if cum_prob > cutoff_prob:
                    break

                probs_idx = probs_idx[:cutoff_len]

        log_probs_idx = [(probs_idx[i][0], np.log(np.exp(probs_idx[i][1]))) for i in range(cutoff_len)]

        for l in prefix_set_prev:
            if l not in prefix_set_next:
                log_probs_b_cur[l] = FLT64_MIN
                log_probs_nb_cur[l] = FLT64_MIN

            for idx in range(cutoff_len):
                c, log_prob_c = log_probs_idx[idx][0], log_probs_idx[idx][1]

                if c == blank_id:
                    # emission prob * (previous b + nb)
                    log_probs_prev = log_sum_exp(log_probs_b_prev[l], log_probs_nb_prev[l])
                    log_probs_b_cur[l] = log_sum_exp(log_probs_b_cur[l], log_prob_c + log_probs_prev)

                else:
                    last_char = l[-1]
                    new_char = vocabulary[c]
                    l_plus = l + '_' + new_char

                    if l_plus not in prefix_set_next:
                        log_probs_b_cur[l_plus] = FLT64_MIN
                        log_probs_nb_cur[l_plus] = FLT64_MIN

                    if new_char == last_char:
                        # emission prob * previous nb
                        log_probs_nb_cur[l_plus] = log_sum_exp(log_probs_nb_cur[l_plus], log_prob_c + log_probs_nb_prev[l])

                    else:
                        # emission prob * (previous b + nb)
                        log_probs_prev = log_sum_exp(log_probs_b_prev[l], log_probs_nb_prev[l])
                        log_probs_nb_cur[l_plus] = log_sum_exp(log_probs_nb_cur[l_plus], log_prob_c + log_probs_prev)

                    # add l_plus into prefix_set_next
                    prefix_set_next[l_plus] = log_sum_exp(log_probs_nb_cur[l_plus], log_probs_b_cur[l_plus])

            # add l into prefix_set_next
            prefix_set_next[l] = log_sum_exp(log_probs_b_cur[l], log_probs_nb_cur[l])
        # update probs
        log_probs_b_prev = log_probs_b_cur
        log_probs_nb_prev = log_probs_nb_cur

        # store stop beam_size prefixes
        prefix_set_prev = sorted(prefix_set_next.items(), key=lambda asd: asd[1], reverse=True)

        if beam_size < len(prefix_set_prev):
            prefix_set_prev = prefix_set_prev[:beam_size]

        prefix_set_prev = dict(prefix_set_prev)

    beam_result = []
    for (seq, log_prob) in prefix_set_prev.items():
        if log_prob > FLT64_MIN:
            result = seq[1:]
            beam_result.append([log_prob, result])

    beam_result = sorted(beam_result, key=lambda asd: asd[1], reverse=True)

    return beam_result


def ctc_beam_search_decoder(probs_seq,
                            beam_size,
                            vocabulary,
                            blank_id,
                            cutoff_prob=1.0):
    """
    :param probs_seq: 2-D list of probability distributions over each time
                      step, with each element being a list of normalized
                      probabilities over vocabulary and blank.
    :param beam_size: Width for beam search.
    :param vocabulary: Vocabulary list.
    :param blank_id: ID of blank.
    :param cutoff_prob: Cutoff probability in pruning,
                        default 1.0, no pruning.
    :return: List of tuples of log probability and sentence as decoding
             results, in descending order of the probability.
    """
    # dimension check
    for prob_list in probs_seq:
        if not len(prob_list) == len(vocabulary) + 1:
            raise ValueError("The shape of prob_seq does not match with the "
                             "shape of the vocabulary.")

    # blank_id check
    if not blank_id < len(probs_seq[0]):
        raise ValueError("blank_id shouldn't be greater than probs dimension")

    ## initialize
    # prefix_set_prev: the set containing selected prefixes
    # probs_b_prev: prefixes' probability ending with blank in previous step
    # probs_nb_prev: prefixes' probability ending with non-blank in previous step
    prefix_set_prev = {'\t': 1.0}
    # two states : b, nb && b + nb = 1
    probs_b_prev, probs_nb_prev = {'\t': 1.0}, {'\t': 0.0}

    ## extend prefix in loop
    for time_step in range(len(probs_seq)):
        # prefix_set_next: the set containing candidate prefixes
        # probs_b_cur: prefixes' probability ending with blank in current step
        # probs_nb_cur: prefixes' probability ending with non-blank in current step
        prefix_set_next, probs_b_cur, probs_nb_cur = {}, {}, {}

        prob_idx = list(enumerate(probs_seq[time_step]))
        cutoff_len = len(prob_idx)
        # If pruning is enabled
        if cutoff_prob < 1.0:
            prob_idx = sorted(prob_idx, key=lambda asd: asd[1], reverse=True)
            cutoff_len, cum_prob = 0, 0.0
            for i in range(len(prob_idx)):
                cum_prob += prob_idx[i][1]
                cutoff_len += 1
                if cum_prob >= cutoff_prob:
                    break
            prob_idx = prob_idx[0:cutoff_len]

        for l in prefix_set_prev:
            if l not in prefix_set_next:
                probs_b_cur[l], probs_nb_cur[l] = 0.0, 0.0

            # extend prefix by travering prob_idx
            for index in range(cutoff_len):
                c, prob_c = prob_idx[index][0], prob_idx[index][1]

                if c == blank_id:
                    # emission prob * (previous b + nb)
                    probs_b_cur[l] = prob_c * (probs_b_prev[l] + probs_nb_prev[l])
                else:
                    last_char = l[-1]
                    new_char = vocabulary[c]
                    l_plus = l + '_' + new_char
                    if l_plus not in prefix_set_next:
                        probs_b_cur[l_plus], probs_nb_cur[l_plus] = 0.0, 0.0

                    if new_char == last_char:
                        # emission prob * previous nb
                        probs_nb_cur[l_plus] = prob_c * probs_nb_prev[l]
                    else:
                        # emission prob * (previous b + nb)
                        probs_nb_cur[l_plus] = prob_c * (probs_b_prev[l] + probs_nb_prev[l])

                    prefix_set_next[l_plus] = probs_nb_cur[l_plus] + probs_b_cur[l_plus]
            # add l into prefix_set_next
            prefix_set_next[l] = probs_b_cur[l] + probs_nb_cur[l]
        # update probs
        probs_b_prev, probs_nb_prev = probs_b_cur, probs_nb_cur

        ## store top beam_size prefixes
        prefix_set_prev = sorted(
            prefix_set_next.items(), key=lambda asd: asd[1], reverse=True)
        if beam_size < len(prefix_set_prev):
            prefix_set_prev = prefix_set_prev[:beam_size]
        prefix_set_prev = dict(prefix_set_prev)

    beam_result = []
    for seq, prob in prefix_set_prev.items():
        if prob > 0.0 and len(seq) > 1:
            result = seq[1:]
            # score last word by external scorer
            if prob > 0.0:
                beam_result.append((prob, result))

    ## output top beam_size decoding results
    beam_result = sorted(beam_result, key=lambda asd: asd[0], reverse=True)

    return beam_result
