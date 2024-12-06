import numpy
import os

from utf_utils import transform_german_word

def load_lancaster_en_de_it(args):
    print('loading original lancaster ratings...')
    lancaster_ratings = {
                         'en' : read_lancaster_ratings(),
                         }
    trans_from_en = dict()
    ### german translation
    print('loading translations of ratings...')
    for lang in ['de', 'it']:
        print(lang)
        lancaster_ratings[lang] = dict()
        trans_from_en[lang] = dict()
        missing_words = list()
        with open(os.path.join('data', 'translations', 'lanc_fern_{}_to_en.tsv'.format(lang))) as i:
            for l in i:
                line = l.strip().split('\t')
                if lang == 'de':
                    #if 'frequency' in args.model:
                    #    ws = line[0].lower()
                    #else:
                    #    ws = transform_german_word(line[0].lower())
                    ws = transform_german_word(line[0].lower())
                    for w in ws:
                        ### ratings
                        try:
                            lancaster_ratings['de'][w] = lancaster_ratings['en'][line[1]]
                        except KeyError:
                            #print(w)
                            missing_words.append(line[1])
                            pass
                        ### translations
                        try:
                            trans_from_en['de'][line[1]].add(w)
                        except KeyError:
                            trans_from_en['de'][line[1]] = {w}
                elif lang == 'it':
                    for w in [line[0].lower(), line[0].capitalize()]:
                        ### ratings
                        try:
                            lancaster_ratings['it'][w] = lancaster_ratings['en'][line[1]]
                        except KeyError:
                            missing_words.append(line[1])
                            pass
                        ### translations
                        try:
                            trans_from_en['it'][line[1]].add(w)
                        except KeyError:
                            trans_from_en['it'][line[1]] = {w}
        print('missing words from the lancaster norms: {}'.format(set(missing_words)))
    return lancaster_ratings, trans_from_en

def read_lancaster_ratings():
    norms = dict()
    ### sensory ratings
    file_path = os.path.join(
                             'data',
                             'psycholinguistic_norms',
                             'lancaster_sensorimotor',
                             'Lancaster_sensorimotor_norms_for_39707_words.tsv',
                             )
    assert os.path.exists(file_path)
    with open(file_path) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w.lower() for w in line]
                relevant_keys = [w.lower() for w in line if '.mean' in w or w=='Minkowski3.perceptual']
                continue
            assert len(line) == len(header)
            word = line[0].strip().lower()
            marker = False
            for k in relevant_keys:
                if 'inkow' in k:
                    continue
                try:
                    assert float(line[header.index(k)]) <= 5 
                except AssertionError:
                    ### correcting for a silly mistake in the ratings...
                    line[header.index(k)] = '.{}'.format(line[header.index(k)])
            if word not in norms.keys():
                norms[word] = dict()
            for k in relevant_keys:
                var = k.split('.')[0]
                val = float(line[header.index(k)])
                ### minimum is 0, max is 5
                if 'inkow' not in k:
                    assert val >= 0. and val <= 5.
                    curr_val = float(val) / 5
                else:
                    curr_val = float(val)
                norms[word][var] = curr_val

    return norms

def read_brysbaert_conc_ratings():
    ### concreteness
    norms = {'concreteness' : dict()}
    with open(os.path.join(
                           'data', 
                           'psycholinguistic_norms', 
                           'brysbaert_concreteness', 
                           'Concreteness_ratings_Brysbaert_et_al_BRM.txt')
                           ) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            ### minimum is 1, max is 5
            val = float(line[2])
            assert val >= 1. and val <= 5.
            curr_val = (val - 1) / (5 - 1)
            w = line[0].lower().strip()
            if w in norms['visual'].keys():
                norms['concreteness'][w] = curr_val
    return norms
