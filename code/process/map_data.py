#! python 
# @Time    : 17-9-7
# @Author  : kay
# @File    : map_data.py
# @E-mail  : 861186267@qq.com
# @Function:

def get_corpus(path):
    """
    :param path: 
    :return: 
    """
    fcorpus = open(path, 'r')
    lines = fcorpus.readlines()

    corpus_words = {}

    for line in lines:
        line = line.strip('\n').strip(' ')
        items = line.split(' ')

        for item in items:
            if item not in corpus_words:
                print('not in')
                print('item:', item)
                corpus_words[item] = 1
            else:
                print('in')
                print('item:', item)
                corpus_words[item] += 1

    return corpus_words


def get_dictionary(path):
    """
    :param path: 
    :return: 
    """
    fdict = open(path, 'r')
    dict_lines = fdict.readlines()
    dict_words = {}

    for line in dict_lines:
        line = line.strip('\n').strip(' ')
        items = line.split(' ')

        if items[0] not in dict_words:
            # print('not in dict_words')
            dict_words[items[0]] = ''
            for i in range(2, len(items)):
                dict_words[items[0]] += ' ' + items[i]
        # else:
        #     print('in dict_words, item[0]:', items[0])
        fdict.close()

    return dict_words


def get_phonemes(path):
    """
    :param path: 
    :return: 
    """
    fphos = open(path, 'r')
    lines = fphos.readlines()
    phonemes = []

    for line in lines:
        line = line.strip('\n')
        items = line.split(' ')
        print(items)
        item = items[0].split('_')[0]

        if item not in phonemes:
            print('false')
            phonemes.append(item)
        else:
            print('true')

        fphos.close()

    return phonemes


def get_phn():
    # 138 phonemes
    phn = ['sil', 'A1', 'A2', 'A3', 'A4', 'AG1', 'AG2', 'AG3', 'AG4', 'AI1',
           'AI2', 'AI3', 'AI4', 'AN1', 'AN2', 'AN3', 'AN4', 'AO1', 'AO2',
           'AO3', 'AO4', 'B', 'BI', 'BU', 'C', 'CH', 'CHU', 'CU',
           'D', 'DI', 'DU', 'E1', 'E2', 'E3', 'E4', 'EG1', 'EG2',
           'EG3', 'EG4', 'EI1', 'EI2', 'EI3', 'EI4', 'EN1', 'EN2', 'EN3',
           'EN4', 'ER2', 'ER3', 'ER4', 'F', 'G', 'GS', 'GU', 'H',
           'HU', 'I1', 'I2', 'I3', 'I4', 'IE1', 'IE2', 'IE3', 'IE4',
           'IG1', 'IG2', 'IG3', 'IG4', 'IH1', 'IH2', 'IH3', 'IH4', 'IN1',
           'IN2', 'IN3', 'IN4', 'JI', 'JU', 'K', 'KU', 'L', 'LI',
           'LU', 'LYU', 'M', 'MI', 'MU', 'N', 'NI', 'NU', 'NYU',
           'O1', 'O2', 'O3', 'O4', 'OG1', 'OG2', 'OG3', 'OG4', 'OU1',
           'OU2', 'OU3', 'OU4', 'P', 'PI', 'QI', 'QU', 'R', 'RU',
           'S', 'SH', 'SHU', 'SU', 'T', 'TI', 'TU', 'U1', 'U2',
           'U3', 'U4', 'UN1', 'UN2', 'UN3', 'UN4', 'V1', 'V2', 'V3',
           'V4', 'W', 'XI', 'XU', 'Y', 'YU', 'Z', 'ZH', 'ZHU',
           'ZU', 'rr']

    return phn


def get_shrink_phn():
    # merge similar sounds into one (138 to 69 phonemes)
    shrink_phn = ['sil', 'A', 'AG', 'AI', 'AN', 'AO', 'B', 'BI', 'BU',
                  'C', 'CU', 'D', 'DI', 'DU', 'E', 'EG', 'EI', 'EN',
                  'ER', 'F', 'G', 'GS', 'GU', 'H', 'HU', 'I', 'IE',
                  'IG', 'IH', 'IN', 'JI', 'JU', 'K', 'KU', 'L', 'LI',
                  'LU', 'LYU', 'M', 'MI', 'MU', 'N', 'NI', 'NU', 'NYU',
                  'O', 'OG', 'OU', 'P', 'PI', 'QI', 'QU', 'R', 'RU',
                  'S', 'SU', 'T', 'TI', 'TU', 'U', 'UN', 'V', 'W',
                  'XI', 'XU', 'Y', 'YU', 'Z', 'ZU']

    return shrink_phn


def get_initial_phn():
    """
    the initial phonemes
    :return: 
    """
    initial_phn = ['B', 'BI', 'BU', 'C', 'CU', 'D', 'DI', 'DU', 'F', 'G', 'GS',
                   'GU', 'H', 'HU', 'JI', 'JU', 'K', 'KU', 'L', 'LI', 'LU', 'LYU',
                   'M', 'MI', 'MU', 'N', 'NI', 'NU', 'NYU', 'P', 'PI', 'QI', 'QU',
                   'R', 'RU', 'S', 'SU', 'T', 'TI', 'TU', 'W', 'XI', 'XU', 'Y', 'YU', 'Z', 'ZU']

    return initial_phn


def get_rhythm_phn():
    """
    the rhyme phonemes
    :return: 
    """
    rhythm_phn = ['A', 'AG', 'AI', 'AN', 'AO',
                 'E', 'EG', 'EI', 'EN', 'ER',
                 'I', 'IE', 'IG', 'IH', 'IN',
                 'O', 'OG', 'OU', 'U', 'UN',
                 'V']

    return rhythm_phn


def get_mapping_phn():
    mapping_phn = {'sil': 'sil',
                   'A1': 'A', 'A2': 'A', 'A3': 'A', 'A4': 'A',
                   'AG1': 'AG', 'AG2': 'AG', 'AG3': 'AG', 'AG4': 'AG',
                   'AI1': 'AI', 'AI2': 'AI', 'AI3': 'AI', 'AI4': 'AI',
                   'AN1': 'AN', 'AN2': 'AN', 'AN3': 'AN', 'AN4': 'AN',
                   'AO1': 'AO', 'AO2': 'AO', 'AO3': 'AO', 'AO4': 'AO',
                   'B': 'B',
                   'BI': 'BI',
                   'BU': 'BU',

                   # 'C', 'CU', 'D', 'DI', 'DU', 'E', 'EG', 'EI', 'EN',
                   'C': 'C', 'CH': 'C',
                   'CHU': 'CU', 'CU': 'CU',
                   'D': 'D',
                   'DI': 'DI',
                   'DU': 'DU',
                   'E1': 'E', 'E2': 'E', 'E3': 'E', 'E4': 'E',
                   'EG1': 'EG', 'EG2': 'EG', 'EG3': 'EG', 'EG4': 'EG',
                   'EI1': 'EI', 'EI2': 'EI', 'EI3': 'EI', 'EI4': 'EI',
                   'EN1': 'EN', 'EN2': 'EN', 'EN3': 'EN', 'EN4': 'EN',

                   # 'ER', 'F', 'G', 'GS', 'GU', 'H', 'HU', 'I', 'IE',
                   'ER2': 'ER', 'ER3': 'ER', 'ER4': 'ER', 'rr': 'ER',
                   'F': 'F',
                   'G': 'G',
                   'GS': 'GS',
                   'GU': 'GU',
                   'H': 'H',
                   'HU': 'HU',
                   'I1': 'I', 'I2': 'I', 'I3': 'I', 'I4': 'I',
                   'IE1': 'IE', 'IE2': 'IE', 'IE3': 'IE', 'IE4': 'IE',

                   # 'IG', 'IH', 'IN', 'JI', 'JU', 'K', 'KU', 'L', 'LI',
                   'IG1': 'IG', 'IG2': 'IG', 'IG3': 'IG', 'IG4': 'IG',
                   'IH1': 'IH', 'IH2': 'IH', 'IH3': 'IH', 'IH4': 'IH',
                   'IN1': 'IN', 'IN2': 'IN', 'IN3': 'IN', 'IN4': 'IN',
                   'JI': 'JI',
                   'JU': 'JU',
                   'K': 'K',
                   'KU': 'KU',
                   'L': 'L',
                   'LI': 'LI',

                   # 'LU', 'LYU', 'M', 'MI', 'MU', 'N', 'NI', 'NU', 'NYU',
                   'LU': 'LU',
                   'LYU': 'LYU',
                   'M': 'M',
                   'MI': 'MI',
                   'MU': 'MU',
                   'N': 'N',
                   'NI': 'NI',
                   'NU': 'NU',
                   'NYU': 'NYU',

                   # 'O', 'OG', 'OU', 'P', 'PI', 'QI', 'QU', 'R', 'RU',
                   'O1': 'O', 'O2': 'O', 'O3': 'O', 'O4': 'O',
                   'OG1': 'OG', 'OG2': 'OG', 'OG3': 'OG', 'OG4': 'OG',
                   'OU1': 'OU', 'OU2': 'OU', 'OU3': 'OU', 'OU4': 'OU',
                   'P': 'P',
                   'PI': 'PI',
                   'QI': 'QI',
                   'QU': 'QU',
                   'R': 'R',
                   'RU': 'RU',

                   # 'S', 'SU', 'T', 'TI', 'TU', 'U', 'UN', 'V', 'W',
                   'S': 'S', 'SH': 'S',
                   'SHU': 'SU', 'SU': 'SU',
                   'T': 'T',
                   'TI': 'TI',
                   'TU': 'TU',
                   'U1': 'U', 'U2': 'U', 'U3': 'U', 'U4': 'U',
                   'UN1': 'UN', 'UN2': 'UN', 'UN3': 'UN', 'UN4': 'UN',
                   'V1': 'V', 'V2': 'V', 'V3': 'V', 'V4': 'V',
                   'W': 'W',

                   # 'XI', 'XU', 'Y', 'YU', 'Z', 'ZU'
                   'XI': 'XI',
                   'XU': 'XU',
                   'Y': 'Y',
                   'YU': 'YU',
                   'Z': 'Z', 'ZH': 'Z',
                   'ZHU': 'ZU', 'ZU': 'ZU'}

    return mapping_phn


def get_folds():
    """
    :return: 
    """
    # folds = ['40corpus', 'chufang_bingxiang', 'chufang_dianfanbao', 'js_boxdata1_sox_fixed',
    #          'js_boxdata3_sox_fixed', 'js_recordset_20150801_1m', 'js_recordset_20150801_3m',
    #          'js_recordset_20150801_5m', 'kt_dianshi', 'kt_dingdeng', 'kt_kongtiao', 'kt_kongzhiqi',
    #          'kt_shafa', 'speech_in', 'wakeup_box_1m', 'wakeup_box_3m', 'ws_dianshi', 'ws_KT_kt_ds',
    #          'ZTEcorpus']

    # shuffle
    folds = ['40corpus', 'chufang_bingxiang', 'js_boxdata1_sox_fixed', 'ws_dianshi',
             'js_recordset_20150801_1m', 'kt_dianshi', 'chufang_dianfanbao',
             'speech_in', 'wakeup_box_1m', 'ws_KT_kt_ds', 'js_boxdata3_sox_fixed', 'kt_dingdeng',
             'js_recordset_20150801_3m', 'kt_kongtiao', 'js_recordset_20150801_5m', 'wakeup_box_3m',
             'kt_kongzhiqi', 'ZTEcorpus', 'kt_shafa']
    return folds


def is_chinese(uchar):
    """
    confirm the character whether is chinese or not
    :param uchar: 
    :return: 
    """
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':

        return True

    else:

        return False

def get_dengkong_dataset():
    """
    the evaluation dataset
    :return: 
    """

    datasets = ['alading', 'huanyanse', 'mingliangmoshi', 'qingguandeng', 'qingkaideng', 'shendengshendeng',
                'shuiminmoshi', 'tiaoanyidian', 'tiaoliangyidian', 'wenxinmoshi', 'yedengmoshi', 'yuedumoshi']

    return datasets