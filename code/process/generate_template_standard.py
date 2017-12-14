#! python 
# @Time    : 17-12-4
# @Author  : kay
# @File    : generate_template_standard.py
# @E-mail  : 861186267@qq.com
# @Function:

from process.langconv import *
from process.map_data import *
import os
import numpy as np
import shutil

template_dir = '/home/kay/Desktop/ctc/resource/code/python/data/phn/templates_standard'
if os.path.exists(template_dir):
    shutil.rmtree(template_dir)

if not os.path.exists(template_dir):
    os.makedirs(template_dir)

dictionary_path = './data/align_lexicon.txt'
dict_words = get_dictionary(dictionary_path)
print('len(dict_words):', len(dict_words))

mapping_phn = get_mapping_phn()
shrink_phn = get_shrink_phn()

spotting_words = ['你好小智', '神灯神灯', ]

for idx, ele in enumerate(spotting_words):
    items = ele.split(' ')
    print('items:', items)

    template = []
    template_digits = []

    items_len = len(items)

    template.append(mapping_phn['sil'])
    template_digits.append(shrink_phn.index(mapping_phn['sil']))
    for key in items:
        if key in dict_words:
            # print('in dict_words key: ', key)
            if key != '!SIL':
                value = dict_words[key]
                print(value)
                elements = value.strip(' ').split(' ')
                print(elements)
                for atom in elements:
                    template.append(mapping_phn[atom.split('_')[0]])
                    template_digits.append(shrink_phn.index(mapping_phn[atom.split('_')[0]]))
        else:
            # print('not in dict_words key: ', key)
            for it in key:
                # print('not in dict_words it:', it)
                try:
                    value = dict_words[it]
                except KeyError:
                    try:
                        if is_chinese(it):
                            # convert traditional chinese to sample chinese
                            it = Converter('zh-hans').convert(it)
                            print('it:', it)
                            value = dict_words[it]
                        else:
                            value = dict_words[it.lower()]
                    except KeyError:
                        continue

                elements = value.strip(' ').split(' ')
                # print('not in dict_words elements:', elements)
                for atom in elements:
                    template.append(mapping_phn[atom.split('_')[0]])
                    template_digits.append(shrink_phn.index(mapping_phn[atom.split('_')[0]]))

    template.append(mapping_phn['sil'])
    template_digits.append(shrink_phn.index(mapping_phn['sil']))

    template_path = os.path.join(template_dir, str(idx) + '.npy')
    np.save(template_path, template_digits)

    print('template:', template)
    print('template_digits:', template_digits)